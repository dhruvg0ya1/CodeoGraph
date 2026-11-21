import { formatMoney } from "./format";

/** Fetch the cart total from the server and render it. */
export async function loadTotal(cartId: string): Promise<string> {
  const res = await fetch(`/api/carts/${cartId}`);
  const body = await res.json();
  return formatMoney(body.total, "INR");
}

export class CartClient {
  constructor(private base: string) {}

  async add(item: string, qty: number): Promise<void> {
    await fetch(`${this.base}/add`, { method: "POST", body: JSON.stringify({ item, qty }) });
  }
}
