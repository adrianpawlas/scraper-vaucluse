"""Utility functions for the Vaucluse Studios scraper."""

import logging
import time
from typing import Optional

import requests

from src.config import (
    STORE_CURRENCY,
    TARGET_CURRENCIES,
    FALLBACK_EXCHANGE_RATES,
    REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)


def get_exchange_rates(base_currency: str = STORE_CURRENCY) -> dict[str, dict[str, float]]:
    """Fetch exchange rates for the base currency.

    Tries the free exchangerate-api.com first, falls back to hardcoded rates.

    Returns:
        Dict mapping base currency to dict of target currency -> rate.
        E.g., {"AUD": {"AUD": 1.0, "EUR": 0.60, "USD": 0.66}}
    """
    try:
        # Try free API
        url = f"https://api.exchangerate-api.com/v4/latest/{base_currency}"
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            if "rates" in data:
                logger.info(f"Fetched live exchange rates for {base_currency}")
                return {base_currency: data["rates"]}
    except Exception as e:
        logger.warning(f"Failed to fetch exchange rates from API: {e}")

    # Fall back to hardcoded rates
    logger.info(f"Using fallback exchange rates for {base_currency}")
    rates = FALLBACK_EXCHANGE_RATES.get(base_currency, {})
    if base_currency not in rates:
        rates[base_currency] = 1.0
    return {base_currency: rates}


def format_multi_currency_price(
    price_value: Optional[float],
    base_currency: str = STORE_CURRENCY,
    target_currencies: list[str] = None,
    exchange_rates: dict[str, dict[str, float]] = None,
) -> Optional[str]:
    """Format a price value into multi-currency string.

    Formats like: "55.96AUD , 34.20EUR , 37.50USD"
    Each currency pair separated by comma + space + comma.

    Args:
        price_value: The price in the base currency.
        base_currency: The original currency code (e.g., "AUD").
        target_currencies: List of target currency codes. Defaults to TARGET_CURRENCIES.
        exchange_rates: Exchange rates dict. If None, fetched automatically.

    Returns:
        Formatted string like "55.96AUD , 34.20EUR , 37.50USD" or None.
    """
    if price_value is None:
        return None

    if target_currencies is None:
        target_currencies = TARGET_CURRENCIES

    if exchange_rates is None:
        exchange_rates = get_exchange_rates(base_currency)

    rates = exchange_rates.get(base_currency, {})

    formatted_prices = []
    for currency in target_currencies:
        if currency == base_currency:
            converted = price_value
        else:
            rate = rates.get(currency)
            if rate is None:
                continue
            converted = price_value * rate
        formatted_prices.append(f"{converted:.2f}{currency}")

    if not formatted_prices:
        return f"{price_value:.2f}{base_currency}"

    return " , ".join(formatted_prices)
