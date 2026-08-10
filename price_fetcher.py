"""
Real-time price fetcher for HOVERAir products.

Pulls live prices from HOVERAir's Shopify store JSON API (no auth needed).
Caches results for 1 hour to avoid hammering the API on every request.
"""

import time
import requests

CACHE_TTL = 3600  # 1 hour

PRODUCT_ENDPOINTS = {
    "x1":       "https://us.hoverair.com/products/hoverair-x1-self-flying-camera-combo.json",
    "x1_pro":   "https://us.hoverair.com/products/hoverair-x1-pro-promax.json",
    "aqua":     "https://global.hoverair.com/products/hoverair-aqua.json",
}

_cache = {}  # { key: (timestamp, text) }


def _fetch_product(url: str) -> list[dict]:
    resp = requests.get(url, timeout=8)
    resp.raise_for_status()
    return resp.json()["product"]["variants"]


def _format_x1(variants: list[dict]) -> str:
    lines = ["HOVERAir X1 current prices (USD):"]
    for v in variants:
        if v.get("price_currency") == "USD":
            title = v["title"].replace("Black / ", "").replace("White / ", "")
            price = v["price"]
            compare = v.get("compare_at_price")
            if compare and float(compare) > float(price):
                lines.append(f"  - {title}: ${price} (was ${compare})")
            else:
                lines.append(f"  - {title}: ${price}")
    return "\n".join(lines)


def _format_x1_pro(variants: list[dict]) -> str:
    seen = set()
    lines = ["HOVERAir X1 PRO & PROMAX current prices (USD):"]
    for v in variants:
        if v.get("price_currency") == "USD":
            title = v["title"]
            price = v["price"]
            compare = v.get("compare_at_price")
            if title not in seen:
                seen.add(title)
                if compare and float(compare) > float(price):
                    lines.append(f"  - {title}: ${price} (was ${compare})")
                else:
                    lines.append(f"  - {title}: ${price}")
    return "\n".join(lines)


def _format_aqua(variants: list[dict]) -> str:
    seen = set()
    lines = ["HOVERAir AQUA current prices (USD):"]
    for v in variants:
        if v.get("price_currency") == "USD":
            # Strip country prefix like "CA / Standard" → "Standard"
            bundle = v["title"].split(" / ", 1)[-1] if " / " in v["title"] else v["title"]
            price = v["price"]
            compare = v.get("compare_at_price")
            if bundle not in seen:
                seen.add(bundle)
                if compare and float(compare) > float(price):
                    lines.append(f"  - {bundle}: ${price} (was ${compare})")
                else:
                    lines.append(f"  - {bundle}: ${price}")
    return "\n".join(lines)


_formatters = {
    "x1":     _format_x1,
    "x1_pro": _format_x1_pro,
    "aqua":   _format_aqua,
}


def get_live_prices() -> str:
    """
    Return a formatted string of live prices for all HOVERAir products.
    Results are cached for 1 hour. Falls back gracefully on network errors.
    """
    now = time.time()
    blocks = []

    for key, url in PRODUCT_ENDPOINTS.items():
        cached_at, cached_text = _cache.get(key, (0, None))
        if cached_text and (now - cached_at) < CACHE_TTL:
            blocks.append(cached_text)
            continue
        try:
            variants = _fetch_product(url)
            text = _formatters[key](variants)
            _cache[key] = (now, text)
            blocks.append(text)
        except Exception:
            if cached_text:
                blocks.append(cached_text)  # serve stale on error

    return "\n\n".join(blocks) if blocks else ""


PRICE_KEYWORDS = {
    "price", "cost", "how much", "pricing", "expensive", "cheap", "afford",
    "bundle", "combo", "deal", "offer", "discount", "sale", "buy", "purchase",
    "dollars", "usd", "$", "worth",
}


def is_price_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in PRICE_KEYWORDS)
