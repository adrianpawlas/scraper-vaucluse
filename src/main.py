"""Main orchestration script for the Vaucluse Studios scraper.

This script:
1. Fetches all products from Vaucluse Studios via the Shopify API
2. Downloads product images and generates SigLIP embeddings
3. Generates text embeddings from product info
4. Upserts everything into Supabase
"""

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from src.config import BATCH_SIZE, SOURCE
from src.database import get_supabase_client, get_existing_product_urls, upsert_products_batch
from src.embedder import (
    generate_embeddings_for_product,
)
from src.scraper import get_all_products

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("scraper-vaucluse")


def process_single_product(product: dict, skip_existing: bool = False, existing_urls: set = None) -> Optional[tuple]:
    """Process a single product: generate embeddings, return data for upload.

    Args:
        product: Parsed product dict.
        skip_existing: If True, skip products already in the database.
        existing_urls: Set of product URLs already in the database.

    Returns:
        Tuple of (product, image_embedding, info_embedding) or None if skipped.
    """
    product_url = product.get("product_url")

    # Skip if already exists
    if skip_existing and existing_urls and product_url in existing_urls:
        logger.info(f"Skipping existing product: {product['title']}")
        return None

    logger.info(f"Processing: {product['title']} ({product['id']})")

    try:
        # Generate embeddings
        image_emb, info_emb = generate_embeddings_for_product(product)
        return (product, image_emb, info_emb)
    except Exception as e:
        logger.error(f"Failed to process product {product['id']}: {e}")
        return None


def run_scraper(
    skip_existing: bool = True,
    max_workers: int = 3,
    max_products: Optional[int] = None,
):
    """Run the full scraper pipeline.

    Args:
        skip_existing: Skip products already in the database.
        max_workers: Maximum number of concurrent workers for embedding generation.
        max_products: Maximum number of products to process (for testing).
    """
    start_time = time.time()
    logger.info("=" * 60)
    logger.info(f"Starting Vaucluse Studios Scraper ({SOURCE})")
    logger.info("=" * 60)

    # Step 1: Fetch all products from Shopify
    logger.info("\n📡 Step 1: Fetching products from Shopify API...")
    products = get_all_products()

    if not products:
        logger.error("No products found! Aborting.")
        sys.exit(1)

    logger.info(f"✅ Found {len(products)} products total.")

    # Limit products if specified
    if max_products and max_products < len(products):
        products = products[:max_products]
        logger.info(f"Limited to {max_products} products for testing.")

    # Step 2: Check existing products in database
    existing_urls = set()
    if skip_existing:
        logger.info("\n🔍 Step 2: Checking existing products in database...")
        try:
            client = get_supabase_client()
            existing_urls = get_existing_product_urls(client)
            logger.info(f"✅ Found {len(existing_urls)} existing products in database.")
        except Exception as e:
            logger.warning(f"Could not check existing products: {e}. Will process all.")
            existing_urls = set()

    # Step 3: Generate embeddings and prepare data
    logger.info("\n🧠 Step 3: Generating embeddings (image + text)...")

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_product, p, skip_existing, existing_urls): p
            for p in products
        }

        completed = 0
        total = len(futures)
        for future in as_completed(futures):
            completed += 1
            product = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                logger.info(f"Progress: {completed}/{total} products processed")
            except Exception as e:
                logger.error(f"Error processing {product['title']}: {e}")

    logger.info(f"✅ Generated embeddings for {len(results)}/{total} products.")

    # Step 4: Upsert to Supabase
    logger.info("\n💾 Step 4: Uploading to Supabase...")

    products_to_upload = [r[0] for r in results]
    embeddings = [(r[1], r[2]) for r in results]

    success_count, fail_count = upsert_products_batch(products_to_upload, embeddings)

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("📊 SCRAPE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Total products found:     {len(products)}")
    logger.info(f"   Products processed:       {len(results)}")
    logger.info(f"   Successfully uploaded:    {success_count}")
    logger.info(f"   Failed:                   {fail_count}")
    logger.info(f"   Time elapsed:             {elapsed:.1f}s")
    logger.info("=" * 60)

    return {
        "total": len(products),
        "processed": len(results),
        "uploaded": success_count,
        "failed": fail_count,
        "elapsed_seconds": elapsed,
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Vaucluse Studios Product Scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full scrape all products
  python -m src.main

  # Scrape only 5 products (for testing)
  python -m src.main --max-products 5

  # Reprocess all products (ignore existing)
  python -m src.main --no-skip-existing

  # Slower, uses fewer workers
  python -m src.main --workers 2
        """,
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip products already in the database (default: True)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help=f"Number of concurrent workers for embeddings (default: 3)",
    )
    parser.add_argument(
        "--max-products",
        type=int,
        default=None,
        help="Maximum number of products to process (for testing)",
    )

    args = parser.parse_args()

    run_scraper(
        skip_existing=args.skip_existing,
        max_workers=args.workers,
        max_products=args.max_products,
    )


if __name__ == "__main__":
    main()
