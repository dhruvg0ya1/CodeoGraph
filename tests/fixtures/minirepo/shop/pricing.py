"""Prices, discounts and tax."""

TAX_RATE = 0.18


def apply_discount(price: float, percent: float) -> float:
    """Return the price after a percentage discount, never below zero."""
    if percent < 0 or percent > 100:
        raise ValueError("percent must be between 0 and 100")
    return max(0.0, price * (1 - percent / 100))


def add_tax(price: float) -> float:
    """Add GST at TAX_RATE."""
    return round(price * (1 + TAX_RATE), 2)


def final_price(price: float, percent: float = 0) -> float:
    """Discount first, then tax."""
    return add_tax(apply_discount(price, percent))
