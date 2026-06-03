"""Database module for Supabase operations with smart upsert, change detection, and stale management."""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client

from src.config import (
    SUPABASE_URL,
    SUPABASE_KEY,
    SUPABASE_TABLE,
    SOURCE,
    BATCH_UPSERT_SIZE,
    STALE_MISSED_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Fields that determine whether a product has "changed"
_COMPARISON_FIELDS = [
    "title", "description", "category", "price", "sale",
    "image_url", "additional_images", "size", "tags", "brand",
    "gender", "second_hand", "metadata",
]

_FAILED_LOG_PATH = "failed_products.log"


def get_supabase_client():
    """Create and return a Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _log_failed_product(product: dict, reason: str):
    """Log a failed product to the local log file."""
    try:
        with open(_FAILED_LOG_PATH, "a") as f:
            f.write(f"[{datetime.now(timezone.utc).isoformat()}] {reason}: {json.dumps(product, ensure_ascii=False)}\n")
    except Exception as e:
        logger.error(f"Failed to write to error log: {e}")


def fetch_all_products(client) -> dict[str, dict]:
    """Fetch all existing products for our source from Supabase.

    Returns:
        Dict mapping product_url -> full row dict.
    """
    all_rows = {}
    offset = 0
    limit = 1000

    while True:
        try:
            response = (
                client.table(SUPABASE_TABLE)
                .select("*")
                .eq("source", SOURCE)
                .range(offset, offset + limit - 1)
                .execute()
            )
            if not response.data:
                break
            for row in response.data:
                url = row.get("product_url")
                if url:
                    all_rows[url] = row
            if len(response.data) < limit:
                break
            offset += limit
        except Exception as e:
            logger.error(f"Failed to fetch products at offset {offset}: {e}")
            break

    logger.info(f"Fetched {len(all_rows)} existing products from database.")
    return all_rows


def has_product_changed(scraped: dict, existing: dict) -> bool:
    """Compare a scraped product against the existing DB record.

    Returns True if any field has changed, False otherwise.
    Normalizes None and empty string to avoid false positives.
    """
    for field in _COMPARISON_FIELDS:
        new_val = scraped.get(field)
        old_val = existing.get(field)

        # Normalize None and empty string to the same representation
        if new_val is None or new_val == "":
            new_val = None
        if old_val is None or old_val == "":
            old_val = None

        # Normalize lists for order-independent comparison
        if isinstance(new_val, list):
            new_val = sorted(new_val)
        if isinstance(old_val, list):
            old_val = sorted(old_val)

        # Compare as JSON strings for nested structures (dicts, lists)
        if isinstance(new_val, (dict, list)):
            new_str = json.dumps(new_val, sort_keys=True, ensure_ascii=False)
            old_str = json.dumps(old_val, sort_keys=True, ensure_ascii=False) if old_val else "null"
            if new_str != old_str:
                logger.debug(f"Field '{field}' changed")
                return True
        else:
            # Simple value comparison, avoid str(None) vs str("") pitfall
            if new_val != old_val:
                logger.debug(f"Field '{field}' changed: {old_val!r} -> {new_val!r}")
                return True

    return False


def build_upsert_row(product: dict, image_embedding=None, info_embedding=None) -> dict:
    """Build a database row from a parsed product dict.

    Sets created_at on every upsert (acts as "last imported" timestamp).
    Strips None values so Supabase uses column defaults.
    """
    row = {
        "id": product["id"],
        "source": SOURCE,
        "product_url": product["product_url"],
        "affiliate_url": product.get("affiliate_url"),
        "image_url": product.get("image_url"),
        "brand": product.get("brand"),
        "title": product.get("title", ""),
        "description": product.get("description"),
        "category": product.get("category"),
        "gender": product.get("gender"),
        "size": product.get("size"),
        "second_hand": product.get("second_hand", False),
        "country": None,  # Always NULL per user request
        "compressed_image_url": product.get("compressed_image_url"),
        "tags": product.get("tags"),
        "other": product.get("other"),
        "price": product.get("price"),
        "sale": product.get("sale"),
        "additional_images": product.get("additional_images"),
        "metadata": product.get("metadata"),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Add embeddings if provided
    if image_embedding is not None:
        row["image_embedding"] = image_embedding
    if info_embedding is not None:
        row["info_embedding"] = info_embedding

    # Remove None values (let Supabase use column defaults)
    return {k: v for k, v in row.items() if v is not None}


def upsert_product_batch(
    client,
    rows: list[dict],
    retries: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Upsert a batch of products into Supabase.

    Uses the unique constraint (source, product_url) for conflict resolution.

    Args:
        client: Supabase client instance.
        rows: List of database row dicts.
        retries: Number of retry attempts on failure.

    Returns:
        Tuple of (succeeded_rows, failed_rows).
    """
    if not rows:
        return [], []

    for attempt in range(retries):
        try:
            response = (
                client.table(SUPABASE_TABLE)
                .upsert(rows, on_conflict="source, product_url")
                .execute()
            )
            if hasattr(response, 'data') and response.data:
                succeeded = response.data
                failed = [r for r in rows if r["product_url"] not in {s.get("product_url") for s in succeeded}]
                return succeeded, failed
            else:
                logger.warning(f"Batch upsert returned no data (attempt {attempt + 1})")
        except Exception as e:
            logger.warning(f"Batch upsert failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    # All retries exhausted
    logger.error(f"Batch upsert failed after {retries} attempts for {len(rows)} products")
    for row in rows:
        _log_failed_product(row, f"Batch upsert failed after {retries} retries")
    return [], rows


def upsert_products_smart(
    client,
    new_products: list[dict],
    updated_products: list[dict],
) -> dict:
    """Upsert new and updated products in batches.

    Each upserted row includes fresh embeddings for new products,
    and may include regenerated embeddings for updated products.

    Args:
        client: Supabase client instance.
        new_products: List of (parsed_product_dict, image_emb, info_emb) tuples.
        updated_products: List of (parsed_product_dict, image_emb, info_emb) tuples.

    Returns:
        Dict with counts: new_added, updated, failed.
    """
    new_added = 0
    updated = 0
    failed = 0

    # Build rows for new products
    new_rows = []
    for product, img_emb, txt_emb in new_products:
        row = build_upsert_row(product, img_emb, txt_emb)
        new_rows.append(row)

    # Build rows for updated products
    updated_rows = []
    for product, img_emb, txt_emb in updated_products:
        row = build_upsert_row(product, img_emb, txt_emb)
        updated_rows.append(row)

    # Upsert new products in batches
    logger.info(f"Upserting {len(new_rows)} new products in batches of {BATCH_UPSERT_SIZE}...")
    for i in range(0, len(new_rows), BATCH_UPSERT_SIZE):
        batch = new_rows[i:i + BATCH_UPSERT_SIZE]
        succeeded, failed_rows = upsert_product_batch(client, batch)
        new_added += len(succeeded)
        failed += len(failed_rows)

    # Upsert updated products in batches
    logger.info(f"Upserting {len(updated_rows)} updated products in batches of {BATCH_UPSERT_SIZE}...")
    for i in range(0, len(updated_rows), BATCH_UPSERT_SIZE):
        batch = updated_rows[i:i + BATCH_UPSERT_SIZE]
        succeeded, failed_rows = upsert_product_batch(client, batch)
        updated += len(succeeded)
        failed += len(failed_rows)

    return {
        "new_added": new_added,
        "updated": updated,
        "failed": failed,
    }


def handle_stale_products(
    client,
    seen_urls: set[str],
    existing_products: dict[str, dict],
) -> dict:
    """Identify and handle stale products.

    Products that were NOT seen this run get their missed_runs counter incremented.
    Products with missed_runs >= threshold get deleted.

    Args:
        client: Supabase client instance.
        seen_urls: Set of product_urls seen in the current scrape run.
        existing_products: Dict of all existing products for our source.

    Returns:
        Dict with counts: missed_incremented, deleted.
    """
    missed_incremented = 0
    deleted = 0

    for url, existing_row in existing_products.items():
        if url in seen_urls:
            continue  # Product was seen this run, all good

        product_id = existing_row.get("id", url)

        # Read current missed_runs from the `other` field
        other_raw = existing_row.get("other")
        try:
            other_data = json.loads(other_raw) if other_raw and isinstance(other_raw, str) else {}
        except (json.JSONDecodeError, TypeError):
            other_data = {}

        missed_runs = other_data.get("missed_runs", 0) + 1
        other_data["missed_runs"] = missed_runs
        other_updated = json.dumps(other_data, ensure_ascii=False)

        if missed_runs >= STALE_MISSED_THRESHOLD:
            # Delete stale product
            try:
                logger.info(f"Deleting stale product (missed {missed_runs}x): {product_id}")
                client.table(SUPABASE_TABLE).delete().eq("source", SOURCE).eq("product_url", url).execute()
                deleted += 1
            except Exception as e:
                logger.error(f"Failed to delete stale product {product_id}: {e}")
        else:
            # Increment missed counter
            try:
                logger.info(f"Marking product as missed (run {missed_runs}/{STALE_MISSED_THRESHOLD}): {product_id}")
                client.table(SUPABASE_TABLE).update({"other": other_updated}).eq("source", SOURCE).eq("product_url", url).execute()
                missed_incremented += 1
            except Exception as e:
                logger.error(f"Failed to update missed counter for {product_id}: {e}")

    return {
        "missed_incremented": missed_incremented,
        "deleted": deleted,
    }


def reset_missed_runs(client, seen_urls: set[str], existing_products: dict[str, dict]):
    """Reset the missed_runs counter for products that were seen this run."""
    for url in seen_urls:
        if url in existing_products:
            existing_row = existing_products[url]
            other_raw = existing_row.get("other")
            try:
                other_data = json.loads(other_raw) if other_raw and isinstance(other_raw, str) else {}
            except (json.JSONDecodeError, TypeError):
                other_data = {}

            if other_data.get("missed_runs", 0) > 0:
                other_data["missed_runs"] = 0
                other_updated = json.dumps(other_data, ensure_ascii=False)
                try:
                    client.table(SUPABASE_TABLE).update({"other": other_updated}).eq("source", SOURCE).eq("product_url", url).execute()
                except Exception as e:
                    logger.warning(f"Failed to reset missed_runs for {url}: {e}")
