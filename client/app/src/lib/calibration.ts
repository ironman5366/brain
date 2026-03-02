export interface ChannelCalibration {
  name: string;
  wire_color: string;
  pin: string;
  // Impedance (from check-impedance)
  impedance_ohms?: number;
  impedance_kohms?: number;
  impedance_rating?: "good" | "ok" | "bad";
  // Signal quality (from check-signal)
  rms_uv?: number;
  line_noise_db?: number;
  dc_drift_uv?: number;
  has_alpha?: boolean;
  alpha_power_ratio?: number;
  signal_rating?: "good" | "ok" | "bad";
  issues?: string[];
  // PSD data for mini spectrum chart
  psd_frequencies?: number[];
  psd_db?: number[];
}

export interface CalibrationState {
  channels: ChannelCalibration[];
  messages: string[];
  impedanceChecked: boolean;
  signalChecked: boolean;
  allGood: boolean;
}

// Wire color name → CSS color for rendering
export const WIRE_CSS_COLORS: Record<string, string> = {
  grey: "#999",
  purple: "#9b59b6",
  blue: "#3498db",
  green: "#2ecc71",
  yellow: "#f1c40f",
  orange: "#e67e22",
  red: "#e74c3c",
  brown: "#8B4513",
  black: "#555",
  unknown: "#666",
};

export type QualityRating = "good" | "ok" | "bad";

export const RATING_COLORS: Record<QualityRating, string> = {
  good: "#4caf50",
  ok: "#ff9800",
  bad: "#f44336",
};
