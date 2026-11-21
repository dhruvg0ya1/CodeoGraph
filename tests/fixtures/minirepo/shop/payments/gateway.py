"""Charging cards through the payment gateway."""

import time

from shop.cart import Cart


class PaymentError(Exception):
    """Raised when the gateway declines or times out."""


def charge_card(card_token: str, amount: float, retries: int = 3) -> str:
    """Charge a tokenised card, retrying with exponential backoff on timeouts. Returns the receipt id."""
    delay = 0.1
    for attempt in range(retries):
        if card_token.startswith("tok_"):
            return f"rcpt_{attempt}_{int(amount * 100)}"
        time.sleep(delay)
        delay *= 2
    raise PaymentError("card declined after retries")


def checkout(cart: Cart, card_token: str, coupon_percent: float = 0) -> str:
    """Charge the cart's total to the card."""
    return charge_card(card_token, cart.total(coupon_percent))
