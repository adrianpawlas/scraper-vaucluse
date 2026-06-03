"""Main orchestration script for the Vaucluse Studios scraper.

Smart pipeline:
1. Fetches all products from Shopify API
2. Compares against existing database records
3. Only generates embeddings for new products or when image changed
4. Batch upserts (50/batch) with retry logic
5. Handles stale products (delete after 2 consecutive misses)
6. Prints detailed run summary
"""

import argparse
import logging
import sys
import time
from typing import Optional

from src.config import SOURCE, EMBEDDING_DELAY, MAX_CONCURRENT_WORKERS
from src.database import (
    get_supabase_client,
    fetch_all_products,
    has_product_changed,
    upsert_products_smart,
    handle_stale_products,
    reset_missed_runs,
)
from src.embedder import generate_embeddings_for_product, generate_text_embedding
from src.scraper import get_all_products

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scraper-vaucluse")


def run_scraper(
    max_products: Optional[int] = None,
):
    """Run the full smart scraper pipeline."""
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"Starting Vaucluse Studios Scraper ({SOURCE})")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Fetch all products from Shopify API
    # ------------------------------------------------------------------
    logger.info("\n📡 Step 1: Fetching products from Shopify API...")
    scraped_products = get_all_products()

    if not scraped_products:
        logger.error("No products found! Aborting.")
        sys.exit(1)

    logger.info(f"✅ Found {len(scraped_products)} products from Shopify.")

    if max_products and max_products < len(scraped_products):
        scraped_products = scraped_products[:max_products]
        logger.info(f"Limited to {max_products} products for testing.")

    # ------------------------------------------------------------------
    # Step 2: Fetch existing products from Supabase
    # ------------------------------------------------------------------
    logger.info("\n🔍 Step 2: Fetching existing products from database...")
    client = get_supabase_client()
    existing_products = fetch_all_products(client)
    logger.info(f"✅ Found {len(existing_products)} products in database.")

    # ------------------------------------------------------------------
    # Step 3: Classify products and generate embeddings where needed
    # ------------------------------------------------------------------
    logger.info("\n🧠 Step 3: Classifying products and generating embeddings...")

    new_products = []       # (parsed_dict, image_emb, info_emb)
    updated_products = []   # (parsed_dict, image_emb, info_emb)
    unchanged_products = []  # parsed_dict only (no embeddings needed)

    seen_urls = set()

    for idx, product in enumerate(scraped_products, 1):
        product_url = product["product_url"]
        seen_urls.add(product_url)
        title = product["title"]

        existing = existing_products.get(product_url)

        if existing is None:
            # NEW product — generate both embeddings
            logger.info(f"[{idx}/{len(scraped_products)}] NEW: {title}")
            img_emb, info_emb = generate_embeddings_for_product(product)
            new_products.append((product, img_emb, info_emb))
            # Stagger: 0.5s delay between API calls
            time.sleep(EMBEDDING_DELAY)

        elif has_product_changed(product, existing):
            # CHANGED product — check if image changed
            old_image = existing.get("image_url")
            new_image = product.get("image_url")
            image_changed = (old_image != new_image)

            if image_changed:
                logger.info(f"[{idx}/{len(scraped_products)}] UPDATED (image changed): {title}")
                img_emb, info_emb = generate_embeddings_for_product(product)
                time.sleep(EMBEDDING_DELAY)
            else:
                logger.info(f"[{idx}/{len(scraped_products)}] UPDATED (fields changed): {title}")
                # Only regenerate text embedding, keep old image embedding
                info_text = product.get("info_text", "")
                if info_text:
                    info_emb = generate_text_embedding(info_text)
                    time.sleep(EMBEDDING_DELAY)
                else:
                    info_emb = None
                img_emb = existing.get("image_embedding")  # Keep old image embedding

            updated_products.append((product, img_emb, info_emb))

        else:
            # UNCHANGED — skip entirely
            logger.info(f"[{idx}/{len(scraped_products)}] UNCHANGED (skipped): {title}")
            unchanged_products.append(product)

    # ------------------------------------------------------------------
    # Step 4: Batch upsert new + updated products
    # ------------------------------------------------------------------
    logger.info("\n💾 Step 4: Uploading to Supabase...")
    upsert_results = upsert_products_smart(client, new_products, updated_products)

    # ------------------------------------------------------------------
    # Step 5: Handle stale products
    # ------------------------------------------------------------------
    logger.info("\n🧹 Step 5: Handling stale products...")

    # Reset missed_runs for products seen this run
    reset_missed_runs(client, seen_urls, existing_products)

    # Handle stale products (not seen this run)
    stale_results = handle_stale_products(client, seen_urls, existing_products)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("📊 SCRAPE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"   Total from Shopify:        {len(scraped_products)}")
    logger.info(f"   New products added:        {upsert_results['new_added']}")
    logger.info(f"   Products updated:          {upsert_results['updated']}")
    logger.info(f"   Products unchanged (skip): {len(unchanged_products)}")
    logger.info(f"   Upsert failures:           {upsert_results['failed']}")
    logger.info(f"   Stale missed (incremented):{stale_results['missed_incremented']}")
    logger.info(f"   Stale deleted:             {stale_results['deleted']}")
    logger.info(f"   Time elapsed:              {elapsed:.1f}s")
    logger.info("=" * 60)

    return {
        "total_from_shopify": len(scraped_products),
        "new_added": upsert_results["new_added"],
        "updated": upsert_results["updated"],
        "unchanged": len(unchanged_products),
        "upsert_failures": upsert_results["failed"],
        "stale_missed": stale_results["missed_incremented"],
        "stale_deleted": stale_results["deleted"],
        "elapsed_seconds": elapsed,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vaucluse Studios Product Scraper — smart pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full scrape
  python -m src.main

  # Scrape only 5 products (for testing)
  python -m src.main --max-products 5
        """,
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Maximum number of products to process (for testing)",
    )

    args = parser.parse_args()
    run_scraper(max_products=args.max_products)


if __name__ == "__main__":
    main()
