"""Configuration settings for the Vaucluse Studios scraper.

Supports environment variable overrides for sensitive values.
"""

import os

# Supabase Configuration
# Can be overridden via environment variables for CI/CD
SUPABASE_URL = os.environ.get(
    "SUPABASE_URL",
    "https://yqawmzggcgpeyaaynrjk.supabase.co",
)
SUPABASE_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlxYXdtemdnY2dwZXlhYXlucmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NTAxMDkyNiwiZXhwIjoyMDcwNTg2OTI2fQ.XtLpxausFriraFJeX27ZzsdQsFv3uQKXBBggoz6P4D4",
)
SUPABASE_TABLE = "products"

# Scraper Configuration
STORE_DOMAIN = "https://vauclusestudios.com"
PRODUCTS_API_URL = f"{STORE_DOMAIN}/products.json"
PER_PAGE = 250  # Shopify max per page

# Store Constants
SOURCE = "scraper-vaucluse"
BRAND = "Vaucluse Studios"
GENDER = "man"
SECOND_HAND = False

# Model Configuration
EMBEDDING_MODEL = "google/siglip-base-patch16-384"
EMBEDDING_DIM = 768

# Currency Configuration
# The store's base currency is AUD.
# User prefers EUR and USD (EUR highest priority), include AUD as well.
STORE_CURRENCY = "AUD"
TARGET_CURRENCIES = ["EUR", "USD", "AUD"]
# Fallback exchange rates (used if API is unavailable)
FALLBACK_EXCHANGE_RATES = {
    "AUD": {"AUD": 1.0, "EUR": 0.60, "USD": 0.66},
}

# Scraping Settings
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60
MAX_RETRIES = 3
BATCH_SIZE = 10  # Products to process concurrently
