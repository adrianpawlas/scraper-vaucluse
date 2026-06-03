"""Database module for Supabase operations."""

import logging
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client

from src.config import SUPABASE_URL, SUPABASE_KEY, SUPABASE_TABLE, SOURCE

logger = logging.getLogger(__name__)


def get_supabase_client():
    """Create and return a Supabase client."""
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def upsert_product(
    client,
    product: dict,
    image_embedding: Optional[list[float]] = None,
    info_embedding: Optional[list[float]] = None,
) -> bool:
    """Upsert a single product into Supabase.

    Uses the unique constraint (source, product_url) for conflict resolution.
    Sets created_at to current time on each upsert (acts as 'last imported' time).

    Args:
        client: Supabase client instance.
        product: Parsed product dict.
        image_embedding: 768-dim image embedding vector.
        info_embedding: 768-dim text embedding vector.

    Returns:
        True if successful, False otherwise.
    """
    try:
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
            "country": product.get("country"),
            "compressed_image_url": product.get("compressed_image_url"),
            "tags": product.get("tags"),
            "other": product.get("other"),
            "price": product.get("price"),
            "sale": product.get("sale"),
            "additional_images": product.get("additional_images"),
            "metadata": product.get("metadata"),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Add embeddings if available
        if image_embedding is not None:
            row["image_embedding"] = image_embedding
        if info_embedding is not None:
            row["info_embedding"] = info_embedding

        # Keep None values out (let Supabase use column defaults)
        row = {k: v for k, v in row.items() if v is not None}

        # Upsert with conflict resolution on (source, product_url)
        response = (
            client.table(SUPABASE_TABLE)
            .upsert(row, on_conflict="source, product_url")
            .execute()
        )

        if hasattr(response, 'data') and response.data:
            logger.info(f"Upserted: {product['title']} ({product['id']})")
            return True
        else:
            logger.warning(f"Upsert returned no data for: {product['title']}")
            return False

    except Exception as e:
        logger.error(f"Failed to upsert {product['id']}: {e}")
        return False


def upsert_products_batch(
    products: list[dict],
    embeddings: list[tuple[Optional[list[float]], Optional[list[float]]]],
) -> tuple[int, int]:
    """Upsert a batch of products into Supabase.

    Args:
        products: List of parsed product dicts.
        embeddings: List of (image_embedding, info_embedding) tuples.

    Returns:
        Tuple of (success_count, fail_count).
    """
    client = get_supabase_client()
    success_count = 0
    fail_count = 0

    for product, (img_emb, txt_emb) in zip(products, embeddings):
        if upsert_product(client, product, img_emb, txt_emb):
            success_count += 1
        else:
            fail_count += 1

    return success_count, fail_count


def get_existing_product_urls(client) -> set[str]:
    """Get all existing product URLs for our source to avoid reprocessing."""
    try:
        all_urls = set()
        offset = 0
        limit = 1000

        while True:
            response = (
                client.table(SUPABASE_TABLE)
                .select("product_url")
                .eq("source", SOURCE)
                .range(offset, offset + limit - 1)
                .execute()
            )
            if not response.data:
                break
            for row in response.data:
                if row.get("product_url"):
                    all_urls.add(row["product_url"])
            if len(response.data) < limit:
                break
            offset += limit

        logger.info(f"Found {len(all_urls)} existing products in database.")
        return all_urls
    except Exception as e:
        logger.error(f"Failed to fetch existing products: {e}")
        return set()
