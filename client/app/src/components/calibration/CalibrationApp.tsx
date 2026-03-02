import { useRef, useEffect, useCallback } from "react";
import { useCalibration } from "../../hooks/useCalibration";
import { useNarrow } from "../../hooks/useMediaQuery";
import type { ChannelCalibration } from "../../lib/calibration";
import { WIRE_CSS_COLORS, RATING_COLORS } from "../../lib/calibration";
import { BAND_COLORS } from "../../lib/bandpower";
import { formatImpedance } from "../../lib/impedance";

// Band boundaries for spectrum shading
const BANDS = [
  { name: "delta", low: 0.5, high: 4 },
  { name: "theta", low: 4, high: 8 },
  { name: "alpha", low: 8, high: 13 },
  { name: "beta", low: 13, high: 30 },
  { name: "gamma", low: 30, high: 60 },
];

const MAX_FREQ = 62;

export function CalibrationApp() {
  const { state, impedanceRunning } = useCalibration();
  const narrow = useNarrow();
  const latestMessage =
    state.messages.length > 0
      ? state.messages[state.messages.length - 1]
      : "Waiting for Claude to begin calibration...";

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      {/* Current instruction */}
      <div
        style={{
          padding: "0.5rem 1rem",
          borderBottom: "1px solid #333",
          backgroundColor: "#1a1a2e",
          fontFamily: "monospace",
          fontSize: "0.8rem",
          color: "#eee",
          lineHeight: 1.5,
          display: "flex",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "0.5rem",
          flexShrink: 0,
        }}
      >
        {latestMessage}
        {impedanceRunning && (
          <span
            style={{
              fontSize: "0.7rem",
              color: "#ff9800",
            }}
          >
            (measuring impedance...)
          </span>
        )}
      </div>

      {/* Main content */}
      <div style={{ flex: 1, display: "flex", flexDirection: "row", overflow: "hidden" }}>
        {/* Channel cards */}
        <div
          style={{
            flex: 1,
            overflow: "auto",
            padding: "0.5rem",
            minWidth: 0,
          }}
        >
          {state.channels.length > 0 ? (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                gap: "0.5rem",
              }}
            >
              {state.channels.map((ch) => (
                <ChannelCard key={ch.name} channel={ch} narrow={narrow} />
              ))}
            </div>
          ) : (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                minHeight: 120,
                fontFamily: "monospace",
                color: "#555",
                fontSize: "0.85rem",
              }}
            >
              No measurements yet. Open this page and tell Claude you're ready
              to calibrate.
            </div>
          )}
        </div>

        {/* Message history sidebar — hidden on narrow since the banner shows the latest */}
        {!narrow && (
          <div
            style={{
              width: "clamp(160px, 20vw, 280px)",
              borderLeft: "1px solid #333",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
              flexShrink: 0,
            }}
          >
            <div
              style={{
                padding: "0.4rem 0.6rem",
                borderBottom: "1px solid #222",
                fontFamily: "monospace",
                fontSize: "0.7rem",
                color: "#666",
                fontWeight: "bold",
              }}
            >
              Instructions
            </div>
            <div
              style={{
                flex: 1,
                overflow: "auto",
                padding: "0.4rem",
                display: "flex",
                flexDirection: "column",
              }}
            >
              {state.messages.map((msg, i) => (
                <div
                  key={i}
                  style={{
                    padding: "0.3rem 0.5rem",
                    marginBottom: "0.3rem",
                    borderLeft: `2px solid ${
                      i === state.messages.length - 1 ? "#7c6fe0" : "#333"
                    }`,
                    fontFamily: "monospace",
                    fontSize: "0.7rem",
                    color:
                      i === state.messages.length - 1 ? "#ccc" : "#666",
                    lineHeight: 1.4,
                  }}
                >
                  {msg}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function ChannelCard({ channel, narrow }: { channel: ChannelCalibration; narrow: boolean }) {
  const wireColor = WIRE_CSS_COLORS[channel.wire_color] ?? "#666";
  const impRating = channel.impedance_rating;
  const sigRating = channel.signal_rating;

  // Worst rating for border accent
  const worstRating = getWorstRating(impRating, sigRating);
  const borderColor = worstRating
    ? RATING_COLORS[worstRating]
    : "#333";

  return (
    <div
      style={{
        backgroundColor: "#1a1a2e",
        border: `1px solid ${borderColor}`,
        borderRadius: 6,
        padding: "0.4rem 0.5rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.25rem",
      }}
    >
      {/* Header: wire color dot + name + ratings inline */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "0.4rem",
          fontFamily: "monospace",
        }}
      >
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            backgroundColor: wireColor,
            border: "1px solid #555",
            flexShrink: 0,
          }}
        />
        <span style={{ fontWeight: "bold", color: "#eee", fontSize: "0.8rem" }}>
          {channel.name}
        </span>
        <span style={{ color: "#666", fontSize: "0.65rem", textTransform: "capitalize" }}>
          {channel.wire_color}
        </span>
        <span style={{ flex: 1 }} />
        {channel.issues && channel.issues.length > 0 && (
          <span style={{ fontFamily: "monospace", fontSize: "0.6rem", color: "#f88" }}>
            {channel.issues.map((issue) => issueLabel(issue)).join(" · ")}
          </span>
        )}
      </div>

      {/* Impedance bar */}
      <ImpedanceRow channel={channel} />

      {/* Mini spectrum */}
      {channel.psd_frequencies && channel.psd_db && (
        <MiniSpectrum
          frequencies={channel.psd_frequencies}
          psd_db={channel.psd_db}
          height={narrow ? 40 : 50}
        />
      )}

      {/* Signal metrics */}
      <SignalMetrics channel={channel} />
    </div>
  );
}

function ImpedanceRow({ channel }: { channel: ChannelCalibration }) {
  const hasData = channel.impedance_ohms != null;
  const rating = channel.impedance_rating;
  const color = rating ? RATING_COLORS[rating] : "#444";
  const barWidth = hasData ? logScale(channel.impedance_ohms!) : 0;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.4rem",
        fontFamily: "monospace",
        fontSize: "0.65rem",
      }}
    >
      <span style={{ color: "#666", width: 14, flexShrink: 0 }}>Z</span>
      <div
        style={{
          flex: 1,
          height: 6,
          backgroundColor: "#222",
          borderRadius: 3,
          overflow: "hidden",
        }}
      >
        {hasData && (
          <div
            style={{
              width: `${barWidth * 100}%`,
              height: "100%",
              backgroundColor: color,
              borderRadius: 3,
              transition: "width 0.3s ease",
            }}
          />
        )}
      </div>
      <span style={{ color: "#aaa", width: 55, textAlign: "right", flexShrink: 0 }}>
        {hasData ? formatImpedance(channel.impedance_ohms!) : "--"}
      </span>
      <RatingBadge rating={rating} />
    </div>
  );
}

function SignalMetrics({ channel }: { channel: ChannelCalibration }) {
  if (channel.rms_uv == null) {
    return (
      <div
        style={{
          fontFamily: "monospace",
          fontSize: "0.6rem",
          color: "#444",
        }}
      >
        Signal: --
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.5rem",
        fontFamily: "monospace",
        fontSize: "0.6rem",
        color: "#888",
        flexWrap: "wrap",
      }}
    >
      <span>{channel.rms_uv} µV</span>
      <span>60Hz: {channel.line_noise_db} dB</span>
      {channel.has_alpha && (
        <span style={{ color: "#22c55e" }}>
          α {Math.round((channel.alpha_power_ratio ?? 0) * 100)}%
        </span>
      )}
      <RatingBadge rating={channel.signal_rating} />
    </div>
  );
}

function RatingBadge({
  rating,
}: {
  rating?: "good" | "ok" | "bad";
}) {
  if (!rating) return null;
  const color = RATING_COLORS[rating];
  return (
    <span
      style={{
        fontFamily: "monospace",
        fontSize: "0.65rem",
        fontWeight: "bold",
        color,
        textTransform: "uppercase",
        width: 30,
        textAlign: "center",
        flexShrink: 0,
      }}
    >
      {rating}
    </span>
  );
}

function MiniSpectrum({
  frequencies,
  psd_db,
  height = 50,
}: {
  frequencies: number[];
  psd_db: number[];
  height?: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || frequencies.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Clear
    ctx.fillStyle = "#111118";
    ctx.fillRect(0, 0, w, h);

    const pad = { left: 4, right: 4, top: 2, bottom: 10 };
    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Y range
    let yMin = Math.floor(Math.min(...psd_db) / 10) * 10;
    let yMax = Math.ceil(Math.max(...psd_db) / 10) * 10;
    if (yMax - yMin < 10) {
      yMin -= 5;
      yMax += 5;
    }
    const yRange = yMax - yMin;

    const freqToX = (f: number) =>
      pad.left + (f / MAX_FREQ) * plotW;
    const dbToY = (db: number) =>
      pad.top + plotH * (1 - (db - yMin) / yRange);

    // Band shading
    for (const band of BANDS) {
      const color = BAND_COLORS[band.name] ?? "#888";
      const x1 = freqToX(band.low);
      const x2 = freqToX(Math.min(band.high, MAX_FREQ));
      ctx.fillStyle = color + "18";
      ctx.fillRect(x1, pad.top, x2 - x1, plotH);
    }

    // 60 Hz line marker
    const line60x = freqToX(60);
    ctx.strokeStyle = "#f4433640";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(line60x, pad.top);
    ctx.lineTo(line60x, pad.top + plotH);
    ctx.stroke();

    // Spectrum line
    if (frequencies.length > 1) {
      ctx.strokeStyle = "#eee";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(freqToX(frequencies[0]), dbToY(psd_db[0]));
      for (let i = 1; i < frequencies.length; i++) {
        ctx.lineTo(freqToX(frequencies[i]), dbToY(psd_db[i]));
      }
      ctx.stroke();
    }

    // X-axis labels
    ctx.fillStyle = "#444";
    ctx.font = "9px monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const f of [10, 20, 30, 40, 50, 60]) {
      ctx.fillText(`${f}`, freqToX(f), pad.top + plotH + 2);
    }
  }, [frequencies, psd_db]);

  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        height,
        borderRadius: 3,
        display: "block",
      }}
    />
  );
}

// Log-scale bar: maps 1kΩ..1MΩ to 0..1
function logScale(ohms: number): number {
  const minLog = Math.log10(1_000);
  const maxLog = Math.log10(1_000_000);
  const val = Math.log10(Math.max(1_000, Math.min(1_000_000, ohms)));
  return (val - minLog) / (maxLog - minLog);
}

function getWorstRating(
  a?: "good" | "ok" | "bad",
  b?: "good" | "ok" | "bad"
): "good" | "ok" | "bad" | undefined {
  const order = { bad: 0, ok: 1, good: 2 };
  if (a == null && b == null) return undefined;
  if (a == null) return b;
  if (b == null) return a;
  return order[a] <= order[b] ? a : b;
}

function issueLabel(issue: string): string {
  switch (issue) {
    case "high_noise":
      return "High noise";
    case "flat_signal":
      return "Flat/disconnected";
    case "high_line_noise":
      return "60Hz interference";
    case "dc_drift":
      return "DC drift";
    default:
      return issue;
  }
}
