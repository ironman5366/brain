export interface ImpedanceThresholds {
  good: number; // upper bound in ohms
  ok: number; // upper bound in ohms
}

export interface ChannelImpedance {
  index: number;
  name: string;
  impedance: number; // ohms
}

export type ImpedanceStatus = "idle" | "running" | "done" | "error";

export type ImpedanceRating = "good" | "ok" | "bad";

export function getImpedanceRating(
  ohms: number,
  thresholds: ImpedanceThresholds
): ImpedanceRating {
  if (ohms < thresholds.good) return "good";
  if (ohms < thresholds.ok) return "ok";
  return "bad";
}

export const IMPEDANCE_COLORS: Record<ImpedanceRating, string> = {
  good: "#4caf50",
  ok: "#ff9800",
  bad: "#f44336",
};

export function formatImpedance(ohms: number): string {
  if (ohms >= 1_000_000) return `${(ohms / 1_000_000).toFixed(1)} MOhm`;
  if (ohms >= 1_000) return `${(ohms / 1_000).toFixed(1)} kOhm`;
  return `${Math.round(ohms)} Ohm`;
}
