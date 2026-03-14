import { useState, useEffect, useRef } from "react";
import type { ControlSignalResponse } from "../../hooks/useControlSignal";

const API_BASE = "http://localhost:8765";
const POLL_MS = 500;
const THRESHOLD = 0.15;

export function AsymmetryApp() {
  const [data, setData] = useState<ControlSignalResponse | null>(null);
  const [instruction, setInstruction] = useState("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    const poll = async () => {
      try {
        const [ctrlRes, instrRes] = await Promise.all([
          fetch(`${API_BASE}/api/control`),
          fetch(`${API_BASE}/api/asymmetry/instruction`),
        ]);
        if (ctrlRes.ok) setData(await ctrlRes.json());
        if (instrRes.ok) {
          const j = await instrRes.json();
          setInstruction(j.text || "");
        }
      } catch {
        // ignore
      }
    };
    poll();
    timerRef.current = setInterval(poll, POLL_MS);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const asym = data?.asymmetry ?? 0;
  const leftAlpha = data
    ? (data.per_channel?.P7?.alpha ?? 0) + (data.per_channel?.O1?.alpha ?? 0)
    : 0;
  const rightAlpha = data
    ? (data.per_channel?.P8?.alpha ?? 0) + (data.per_channel?.O2?.alpha ?? 0)
    : 0;

  const direction =
    asym < -THRESHOLD ? "LEFT" : asym > THRESHOLD ? "RIGHT" : "CENTER";
  const dirColor =
    direction === "LEFT"
      ? "#4fc3f7"
      : direction === "RIGHT"
        ? "#ff8a50"
        : "#888";

  const barPercent = ((asym + 1) / 2) * 100;

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        backgroundColor: "#141416",
        color: "#eee",
        fontFamily: "monospace",
      }}
    >
      {/* Instruction banner */}
      <div
        style={{
          padding: "1rem",
          borderBottom: "1px solid #333",
          backgroundColor: "#1a1a2e",
          fontSize: "1.1rem",
          textAlign: "center",
          minHeight: "3rem",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {instruction || "Asymmetry Check — waiting for instructions..."}
      </div>

      {/* Main area */}
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: "2rem",
          padding: "2rem",
        }}
      >
        {/* Direction label */}
        <div
          style={{
            fontSize: "4rem",
            fontWeight: "bold",
            color: dirColor,
            letterSpacing: "0.2em",
          }}
        >
          {data ? direction : "..."}
        </div>

        {/* Asymmetry bar */}
        <div style={{ width: "100%", maxWidth: 500 }}>
          <div
            style={{
              position: "relative",
              height: 16,
              backgroundColor: "#2a2a3e",
              borderRadius: 8,
              overflow: "visible",
            }}
          >
            {/* Center line */}
            <div
              style={{
                position: "absolute",
                left: "50%",
                top: -4,
                width: 2,
                height: 24,
                backgroundColor: "#555",
              }}
            />
            {/* Indicator */}
            <div
              style={{
                position: "absolute",
                left: `${barPercent}%`,
                top: -2,
                width: 20,
                height: 20,
                borderRadius: "50%",
                backgroundColor: dirColor,
                transform: "translateX(-50%)",
                transition: "left 0.3s ease-out",
                boxShadow: `0 0 12px ${dirColor}`,
              }}
            />
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: 8,
              fontSize: "0.8rem",
              color: "#666",
            }}
          >
            <span>LEFT</span>
            <span>RIGHT</span>
          </div>
        </div>

        {/* Alpha power comparison */}
        <div
          style={{
            display: "flex",
            gap: "4rem",
            fontSize: "1.2rem",
          }}
        >
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#4fc3f7", fontSize: "2rem", fontWeight: "bold" }}>
              {leftAlpha.toFixed(1)}
            </div>
            <div style={{ color: "#888", fontSize: "0.8rem" }}>
              LEFT alpha (P7+O1)
            </div>
          </div>
          <div style={{ textAlign: "center" }}>
            <div style={{ color: "#ff8a50", fontSize: "2rem", fontWeight: "bold" }}>
              {rightAlpha.toFixed(1)}
            </div>
            <div style={{ color: "#888", fontSize: "0.8rem" }}>
              RIGHT alpha (P8+O2)
            </div>
          </div>
        </div>

        {/* Numeric readout */}
        <div style={{ color: "#666", fontSize: "0.85rem" }}>
          asymmetry: {asym.toFixed(3)} &nbsp;|&nbsp; raw: {data?.raw_asymmetry?.toFixed(3) ?? "n/a"} &nbsp;|&nbsp;
          {data?.calibrated ? "calibrated" : `warming up (${data?.update_count ?? 0}/10)`}
        </div>
      </div>

      {/* Per-channel footer */}
      <div
        style={{
          display: "flex",
          gap: "1.5rem",
          padding: "0.5rem 1rem",
          borderTop: "1px solid #222",
          fontSize: "0.7rem",
          color: "#666",
          flexWrap: "wrap",
        }}
      >
        {data?.per_channel &&
          Object.entries(data.per_channel).map(([name, ch]) => (
            <span key={name} style={{ color: ch.rejected ? "#f44" : "#888" }}>
              {name}: a={ch.alpha.toFixed(1)} b={ch.beta.toFixed(1)}
              {ch.rejected ? " [REJ]" : ""}
            </span>
          ))}
      </div>
    </div>
  );
}
