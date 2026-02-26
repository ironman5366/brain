import { useRef, useEffect, useCallback } from "react";
import { useBandPower } from "../../hooks/useBandPower";
import { BAND_COLORS, BAND_NAMES } from "../../lib/bandpower";
import type { BandPowerResponse } from "../../lib/bandpower";

const HISTORY_WINDOW_S = 60;
const PADDING = { top: 20, right: 16, bottom: 28, left: 48 };
const GRID_LINES = [0, 20, 40, 60, 80, 100];

export function BandPowerApp() {
  const { data, history, error } = useBandPower();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(
    (canvas: HTMLCanvasElement) => {
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);

      const w = rect.width;
      const h = rect.height;
      const plotW = w - PADDING.left - PADDING.right;
      const plotH = h - PADDING.top - PADDING.bottom;

      // Background
      ctx.fillStyle = "#141416";
      ctx.fillRect(0, 0, w, h);

      // Grid
      ctx.strokeStyle = "#222";
      ctx.lineWidth = 1;
      ctx.font = "11px monospace";
      ctx.fillStyle = "#555";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";

      for (const pct of GRID_LINES) {
        const y = PADDING.top + plotH * (1 - pct / 100);
        ctx.beginPath();
        ctx.moveTo(PADDING.left, y);
        ctx.lineTo(PADDING.left + plotW, y);
        ctx.stroke();
        ctx.fillText(`${pct}%`, PADDING.left - 6, y);
      }

      // X-axis labels (seconds ago)
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#555";
      for (let s = 0; s <= HISTORY_WINDOW_S; s += 10) {
        const x = PADDING.left + plotW * (1 - s / HISTORY_WINDOW_S);
        ctx.fillText(s === 0 ? "now" : `-${s}s`, x, PADDING.top + plotH + 6);
      }

      if (history.length < 2) return;

      const now = history[history.length - 1].timestamp;

      // Draw each band line
      for (let bi = 0; bi < BAND_NAMES.length; bi++) {
        const bandName = BAND_NAMES[bi];
        ctx.strokeStyle = BAND_COLORS[bandName] ?? "#888";
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        let started = false;
        for (let i = 0; i < history.length; i++) {
          const snap = history[i];
          const age = (now - snap.timestamp) / 1000;
          if (age > HISTORY_WINDOW_S) continue;

          const x = PADDING.left + plotW * (1 - age / HISTORY_WINDOW_S);
          const y =
            PADDING.top + plotH * (1 - snap.relatives[bi] * 100 / 100);

          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        }
        ctx.stroke();
      }
    },
    [history],
  );

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const observer = new ResizeObserver(() => {
      draw(canvas);
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, [draw]);

  // Redraw on data change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas) draw(canvas);
  }, [draw]);

  if (error) {
    return (
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "monospace",
          color: "#f44",
        }}
      >
        Error: {error}
      </div>
    );
  }

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        padding: "1rem",
        gap: "0.75rem",
        minHeight: 0,
      }}
    >
      {/* Chart */}
      <div ref={containerRef} style={{ flex: 1, minHeight: 0 }}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
      </div>

      {/* Legend */}
      <Legend data={data} />
    </div>
  );
}

function Legend({ data }: { data: BandPowerResponse | null }) {
  const bands = data?.bands ?? [];
  const bandMap = new Map(bands.map((b) => [b.name, b]));

  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: "1rem 2rem",
        fontFamily: "monospace",
        fontSize: "0.8rem",
        padding: "0.25rem 0",
      }}
    >
      {BAND_NAMES.map((name) => {
        const band = bandMap.get(name);
        const color = BAND_COLORS[name] ?? "#888";
        return (
          <div
            key={name}
            style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                backgroundColor: color,
                flexShrink: 0,
              }}
            />
            <span style={{ color, fontWeight: "bold", textTransform: "capitalize" }}>
              {name}
            </span>
            {band && (
              <span style={{ color: "#555" }}>
                {band.low}–{band.high} Hz · {band.description} ·{" "}
                <span style={{ color: "#aaa" }}>
                  {(band.relative * 100).toFixed(1)}%
                </span>
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
