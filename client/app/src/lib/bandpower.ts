export interface BandPower {
  name: string;
  low: number;
  high: number;
  power: number;
  relative: number;
  stddev: number;
  description: string;
}

export interface BandPowerResponse {
  bands: BandPower[];
  total_power: number;
  window_samples: number;
  sampling_rate: number;
  num_channels: number;
  error?: string;
}

export interface BandPowerSnapshot {
  timestamp: number;
  relatives: number[]; // parallel to BAND_NAMES order
}

export const BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"] as const;

export const BAND_COLORS: Record<string, string> = {
  delta: "#8b5cf6",
  theta: "#06b6d4",
  alpha: "#22c55e",
  beta: "#f59e0b",
  gamma: "#ef4444",
};
