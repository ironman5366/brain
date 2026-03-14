import { useRef, useEffect, useCallback } from "react";
import { useSpectrum } from "../../hooks/useSpectrum";
import { BAND_COLORS } from "../../lib/bandpower";

const MAX_FREQ = 60; // Hz — covers all EEG bands
const PADDING = { top: 20, right: 16, bottom: 28, left: 56 };

// Band boundaries for shading (capped at MAX_FREQ for display)
const BANDS: { name: string; low: number; high: number }[] = [
  { name: "delta", low: 0.5, high: 4 },
  { name: "theta", low: 4, high: 8 },
  { name: "alpha", low: 8, high: 13 },
  { name: "beta", low: 13, high: 30 },
  { name: "gamma", low: 30, high: MAX_FREQ },
];

export function FFTApp() {
  const { data, error } = useSpectrum();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(
    (canvas: HTMLCanvasElement) => {
      const ctx = canvas.getContext("2d");
      if (!ctx || !data) return;

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

      // Filter data to MAX_FREQ range
      const freqs = data.frequencies;
      const amps = data.amplitudes_db;
      let maxIdx = freqs.length - 1;
      for (let i = 0; i < freqs.length; i++) {
        if (freqs[i] > MAX_FREQ) {
          maxIdx = i;
          break;
        }
      }

      const visibleAmps = amps.slice(0, maxIdx + 1);
      const visibleFreqs = freqs.slice(0, maxIdx + 1);

      // Auto-range Y axis
      let yMin = Math.floor(Math.min(...visibleAmps) / 10) * 10;
      let yMax = Math.ceil(Math.max(...visibleAmps) / 10) * 10;
      if (yMax - yMin < 10) {
        yMin -= 5;
        yMax += 5;
      }
      const yRange = yMax - yMin;

      const freqToX = (f: number) =>
        PADDING.left + (f / MAX_FREQ) * plotW;
      const dbToY = (db: number) =>
        PADDING.top + plotH * (1 - (db - yMin) / yRange);

      // Band shading
      for (const band of BANDS) {
        const color = BAND_COLORS[band.name] ?? "#888";
        const x1 = freqToX(band.low);
        const x2 = freqToX(Math.min(band.high, MAX_FREQ));
        ctx.fillStyle = color + "15"; // very transparent
        ctx.fillRect(x1, PADDING.top, x2 - x1, plotH);
      }

      // Y-axis gridlines
      ctx.strokeStyle = "#222";
      ctx.lineWidth = 1;
      ctx.font = "11px monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#555";

      const yStep = yRange <= 40 ? 5 : 10;
      for (let db = yMin; db <= yMax; db += yStep) {
        const y = dbToY(db);
        ctx.beginPath();
        ctx.moveTo(PADDING.left, y);
        ctx.lineTo(PADDING.left + plotW, y);
        ctx.stroke();
        ctx.fillText(`${db}`, PADDING.left - 6, y);
      }

      // Y-axis label
      ctx.save();
      ctx.translate(12, PADDING.top + plotH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillStyle = "#444";
      ctx.font = "10px monospace";
      ctx.fillText("dB", 0, 0);
      ctx.restore();

      // X-axis labels
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = "#555";
      ctx.font = "11px monospace";
      for (let f = 0; f <= MAX_FREQ; f += 10) {
        const x = freqToX(f);
        ctx.fillText(`${f}`, x, PADDING.top + plotH + 6);
      }

      // X-axis label
      ctx.fillStyle = "#444";
      ctx.font = "10px monospace";
      ctx.fillText("Hz", PADDING.left + plotW + 8, PADDING.top + plotH + 6);

      // Spectrum line
      if (visibleFreqs.length > 1) {
        ctx.strokeStyle = "#eee";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(freqToX(visibleFreqs[0]), dbToY(visibleAmps[0]));
        for (let i = 1; i < visibleFreqs.length; i++) {
          ctx.lineTo(freqToX(visibleFreqs[i]), dbToY(visibleAmps[i]));
        }
        ctx.stroke();
      }
    },
    [data],
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

  if (!data) {
    return (
      <div
        style={{
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontFamily: "monospace",
          color: "#666",
        }}
      >
        Waiting for data...
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
      <BandLegend />
    </div>
  );
}

function BandLegend() {
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
      {BANDS.map((band) => {
        const color = BAND_COLORS[band.name] ?? "#888";
        return (
          <div
            key={band.name}
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
            <span
              style={{
                color,
                fontWeight: "bold",
                textTransform: "capitalize",
              }}
            >
              {band.name}
            </span>
            <span style={{ color: "#555" }}>
              {band.low}–{band.high} Hz
            </span>
          </div>
        );
      })}
    </div>
  );
}
