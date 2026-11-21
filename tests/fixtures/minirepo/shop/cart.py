"""The shopping cart."""

from .pricing import final_price


class Cart:
    """Holds items and computes the total the customer pays."""

    def __init__(self):
        self.items: list[tuple[str, float, int]] = []

    def add(self, name: str, price: float, qty: int = 1) -> None:
        """Add `qty` units of an item."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.items.append((name, price, qty))

    def subtotal(self) -> float:
        return sum(price * qty for _, price, qty in self.items)

    def total(self, coupon_percent: float = 0) -> float:
        """What the customer pays: subtotal, coupon discount, then tax."""
        return final_price(self.subtotal(), coupon_percent)
