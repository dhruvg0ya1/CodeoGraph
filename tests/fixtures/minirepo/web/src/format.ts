/** Format an amount of money with its currency code. */
export function formatMoney(amount: number, currency: string): string {
  return `${currency} ${amount.toFixed(2)}`;
}

export interface Money {
  amount: number;
  currency: string;
}
