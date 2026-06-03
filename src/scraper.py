"""Scraper module for fetching product data from Vaucluse Studios (Shopify)."""

import json
import logging
import re
import time
from typing import Optional

import requests
from bs4 import BeautifulSoup

from src.config import (
    STORE_DOMAIN,
    PRODUCTS_API_URL,
    PER_PAGE,
    SOURCE,
    BRAND,
    GENDER,
    SECOND_HAND,
    STORE_CURRENCY,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
)
from src.utils import get_exchange_rates, format_multi_currency_price

logger = logging.getLogger(__name__)


def fetch_json(url: str, params: Optional[dict] = None, retries: int = MAX_RETRIES) -> Optional[dict]:
    """Fetch JSON from a URL with retry logic."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    }
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed for {url}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def get_all_products_from_api(max_pages: Optional[int] = None) -> list[dict]:
    """Fetch all products from the Shopify API with pagination.

    Stops when an empty page is returned (no more products).
    """
    all_products = []
    page = 1

    while True:
        if max_pages and page > max_pages:
            break

        logger.info(f"Fetching product page {page}...")
        params = {"limit": PER_PAGE, "page": page}
        data = fetch_json(PRODUCTS_API_URL, params=params)

        if not data or "products" not in data:
            logger.warning(f"No data returned for page {page}, stopping.")
            break

        products = data["products"]
        if not products:
            logger.info(f"Page {page} has 0 products. All products scraped!")
            break

        all_products.extend(products)
        logger.info(f"Fetched {len(products)} products from page {page} (total: {len(all_products)})")
        page += 1

    logger.info(f"Total products fetched from API: {len(all_products)}")
    return all_products


def _find_size_option_index(options: list) -> int:
    """Find which option index corresponds to 'Size' (1-based: option1, option2, etc.)."""
    for i, opt in enumerate(options):
        opt_name = (opt.get("name") or "").lower()
        if "size" in opt_name:
            return i + 1
    return 1  # default to option1


def parse_shopify_product(
    shopify_product: dict,
    exchange_rates: Optional[dict] = None,
) -> dict:
    """Parse a Shopify product JSON object into our normalized format.

    Args:
        shopify_product: Raw product dict from the Shopify API.
        exchange_rates: Pre-fetched exchange rates for multi-currency pricing.

    Returns:
        Normalized product dict ready for embedding and database insertion.
    """
    product_id = str(shopify_product["id"])
    handle = shopify_product["handle"]
    title = shopify_product["title"]
    body_html = shopify_product.get("body_html", "") or ""

    # Strip HTML tags for plain text description
    description = BeautifulSoup(body_html, "html.parser").get_text(separator="\n").strip()

    product_type = shopify_product.get("product_type", "") or ""
    # tags from the Shopify API can be a list (newer API) or a comma-separated string
    tags_raw = shopify_product.get("tags", "") or ""
    if isinstance(tags_raw, list):
        tags_list = [t.strip() for t in tags_raw if t and t.strip()]
    elif isinstance(tags_raw, str) and tags_raw:
        tags_list = [t.strip() for t in tags_raw.split(",") if t.strip()]
    else:
        tags_list = []
    vendor = shopify_product.get("vendor", "") or ""
    created_at = shopify_product.get("created_at", "")
    updated_at = shopify_product.get("updated_at", "")

    # Parse variants
    variants = shopify_product.get("variants", [])
    options = shopify_product.get("options", [])

    # Determine which option is the size
    size_option_index = _find_size_option_index(options)
    size_attr = f"option{size_option_index}"

    # Collect sizes and prices from variants
    sizes = set()
    variant_details = []
    first_variant = variants[0] if variants else None

    for v in variants:
        # Extract size from the correct option
        size_val = v.get(size_attr)
        if size_val:
            size_val = size_val.strip()
            if size_val and size_val != "Default Title":
                sizes.add(size_val)

        var_compare = v.get("compare_at_price")
        var_compare_float = float(var_compare) if var_compare else 0

        variant_details.append({
            "id": v["id"],
            "title": v["title"],
            "sku": v.get("sku", ""),
            "option1": v.get("option1"),
            "option2": v.get("option2"),
            "option3": v.get("option3"),
            "price": v.get("price"),
            "compare_at_price": v.get("compare_at_price"),
            "available": v.get("available", True),
        })

    # Determine pricing: price = original, sale = sale price (if any)
    if first_variant:
        var_price = float(first_variant.get("price", 0))
        var_compare = first_variant.get("compare_at_price")
        var_compare_float = float(var_compare) if var_compare else 0

        if var_compare_float > 0:
            original_price_aud = var_compare_float
            sale_price_aud = var_price
        else:
            original_price_aud = var_price
            sale_price_aud = None
    else:
        original_price_aud = None
        sale_price_aud = None

    # Parse images
    images = shopify_product.get("images", [])
    image_urls = [img["src"] for img in images if img.get("src")]

    main_image = image_urls[0] if image_urls else None
    additional_images = image_urls[1:] if len(image_urls) > 1 else []

    # Build product URL
    product_url = f"{STORE_DOMAIN}/products/{handle}"

    # Build category from product_type, splitting on separators
    category = _parse_category(product_type) if product_type else None

    # Format prices in multiple currencies
    price_formatted = format_multi_currency_price(
        original_price_aud,
        base_currency=STORE_CURRENCY,
        exchange_rates=exchange_rates,
    ) if original_price_aud is not None else None

    sale_formatted = format_multi_currency_price(
        sale_price_aud,
        base_currency=STORE_CURRENCY,
        exchange_rates=exchange_rates,
    ) if sale_price_aud is not None else None

    # Build metadata JSON with all product info
    metadata = {
        "shopify_id": product_id,
        "handle": handle,
        "vendor": vendor,
        "product_type": product_type,
        "tags": tags_list,
        "variants": variant_details,
        "options": options,
        "skus": [v["sku"] for v in variant_details if v.get("sku")],
        "created_at": created_at,
        "updated_at": updated_at,
    }

    # Build info text for text embedding (includes all product info)
    sorted_sizes = ", ".join(sorted(sizes)) if sizes else ""
    info_text_parts = [
        f"Title: {title}",
        f"Description: {description}" if description else "",
        f"Category: {category}" if category else "",
        f"Price: {price_formatted}" if price_formatted else "",
        f"Sale: {sale_formatted}" if sale_formatted else "",
        f"Sizes: {sorted_sizes}" if sorted_sizes else "",
        f"Tags: {', '.join(tags_list)}" if tags_list else "",
        f"Brand: {BRAND}",
        f"Gender: {GENDER}",
        f"Product Type: {product_type}" if product_type else "",
    ]
    info_text = ". ".join(filter(None, info_text_parts))

    # Include metadata content in info_text for richer text embedding
    info_text += f" | Metadata: {json.dumps(metadata, ensure_ascii=False)}"

    return {
        "id": handle,
        "source": SOURCE,
        "product_url": product_url,
        "affiliate_url": None,
        "image_url": main_image,
        "brand": BRAND,
        "title": title,
        "description": description if description else None,
        "category": category,
        "gender": GENDER,
        "size": sorted_sizes if sorted_sizes else None,
        "second_hand": SECOND_HAND,
        "country": "AU",
        "compressed_image_url": None,
        "tags": tags_list if tags_list else None,
        "other": None,
        "price": price_formatted,
        "sale": sale_formatted,
        "additional_images": " , ".join(additional_images) if additional_images else None,
        "metadata": json.dumps(metadata, ensure_ascii=False),
        "info_text": info_text,
        "created_at": None,
    }


def _parse_category(product_type: str) -> Optional[str]:
    """Parse category string, splitting on common separators like ' & '.

    E.g., "Sweaters & Hoodies" becomes "Sweaters, Hoodies"
    """
    if not product_type or not product_type.strip():
        return None
    parts = re.split(r'\s*(?:&|,|\|)\s*', product_type)
    parts = [p.strip() for p in parts if p.strip()]
    return ", ".join(parts) if parts else None


def get_all_products() -> list[dict]:
    """Main function to get all products from Vaucluse Studios.

    Fetches all products via the Shopify products.json API with pagination,
    fetches exchange rates for multi-currency pricing, and parses each product.
    """
    logger.info("Starting to scrape all products from Vaucluse Studios...")

    # Fetch exchange rates once for all products
    logger.info(f"Fetching exchange rates (base: {STORE_CURRENCY})...")
    exchange_rates = get_exchange_rates()

    # Fetch products via the main Shopify API
    products = get_all_products_from_api()

    # Parse each product with exchange rates
    parsed_products = [
        parse_shopify_product(p, exchange_rates=exchange_rates)
        for p in products
    ]

    logger.info(f"Successfully parsed {len(parsed_products)} products.")
    return parsed_products
