import type { ImpedanceThresholds } from "../../lib/impedance";
import {
  getImpedanceRating,
  formatImpedance,
  IMPEDANCE_COLORS,
} from "../../lib/impedance";

interface Props {
  name: string;
  impedance: number;
  thresholds: ImpedanceThresholds;
}

// Log-scale bar: maps 1kOhm..1MOhm to 0..1
function logScale(ohms: number): number {
  const minLog = Math.log10(1_000); // 1 kOhm
  const maxLog = Math.log10(1_000_000); // 1 MOhm
  const val = Math.log10(Math.max(1_000, Math.min(1_000_000, ohms)));
  return (val - minLog) / (maxLog - minLog);
}

export function ImpedanceBar({ name, impedance, thresholds }: Props) {
  const rating = getImpedanceRating(impedance, thresholds);
  const color = IMPEDANCE_COLORS[rating];
  const barWidth = logScale(impedance);

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.75rem",
        padding: "0.5rem 0",
      }}
    >
      <span
        style={{
          fontFamily: "monospace",
          fontSize: "0.9rem",
          fontWeight: "bold",
          width: 36,
          textAlign: "right",
          color: "#ccc",
        }}
      >
        {name}
      </span>

      <div
        style={{
          flex: 1,
          height: 20,
          backgroundColor: "#222",
          borderRadius: 4,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${barWidth * 100}%`,
            height: "100%",
            backgroundColor: color,
            borderRadius: 4,
            transition: "width 0.3s ease",
          }}
        />
      </div>

      <span
        style={{
          fontFamily: "monospace",
          fontSize: "0.85rem",
          width: 100,
          textAlign: "right",
          color: "#aaa",
        }}
      >
        {formatImpedance(impedance)}
      </span>

      <span
        style={{
          fontFamily: "monospace",
          fontSize: "0.75rem",
          fontWeight: "bold",
          width: 40,
          textAlign: "center",
          color,
          textTransform: "uppercase",
        }}
      >
        {rating}
      </span>
    </div>
  );
}
