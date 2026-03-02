import { useRef, useEffect, useCallback } from "react";
import {
  useControlSignal,
  type ControlSignalResponse,
} from "../../hooks/useControlSignal";

const BALL_RADIUS = 14;
const TRAIL_LENGTH = 30;
const MAX_SPEED = 300; // pixels per second
const LERP_FACTOR = 0.08; // client-side smoothing between server updates
const GLOW_COLOR = "rgba(124, 111, 224, "; // #7c6fe0 with alpha
const BALL_COLOR = "#9d93e8";

interface BallState {
  x: number;
  y: number;
  targetVx: number;
  targetVy: number;
  smoothVx: number;
  smoothVy: number;
  trail: { x: number; y: number }[];
}

export function BallApp() {
  const { data, error } = useControlSignal();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const ballRef = useRef<BallState | null>(null);
  const dataRef = useRef<ControlSignalResponse | null>(null);
  const rafRef = useRef<number>(0);
  const prevTimeRef = useRef<number>(0);

  // Keep dataRef in sync without triggering re-renders
  dataRef.current = data;

  // Initialize ball state centered in canvas
  const initBall = useCallback((w: number, h: number): BallState => {
    return {
      x: w / 2,
      y: h / 2,
      targetVx: 0,
      targetVy: 0,
      smoothVx: 0,
      smoothVy: 0,
      trail: [],
    };
  }, []);

  // Main render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const loop = (time: number) => {
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Initialize ball if needed
      if (!ballRef.current) {
        ballRef.current = initBall(w, h);
        prevTimeRef.current = time;
      }
      const ball = ballRef.current;
      const dt = Math.min((time - prevTimeRef.current) / 1000, 0.1); // cap at 100ms
      prevTimeRef.current = time;

      // Update target velocity from latest server data
      const d = dataRef.current;
      if (d && d.calibrated) {
        ball.targetVx = d.asymmetry * MAX_SPEED;
        ball.targetVy = -(d.concentration - 0.5) * 2 * MAX_SPEED; // negative: up is less Y
      }

      // Smooth velocity
      ball.smoothVx += LERP_FACTOR * (ball.targetVx - ball.smoothVx);
      ball.smoothVy += LERP_FACTOR * (ball.targetVy - ball.smoothVy);

      // Update position
      ball.x += ball.smoothVx * dt;
      ball.y += ball.smoothVy * dt;

      // Soft clamp (rubber band at boundaries)
      const margin = BALL_RADIUS + 4;
      if (ball.x < margin) ball.x = margin + (ball.x - margin) * 0.3;
      if (ball.x > w - margin) ball.x = w - margin + (ball.x - (w - margin)) * 0.3;
      if (ball.y < margin) ball.y = margin + (ball.y - margin) * 0.3;
      if (ball.y > h - margin) ball.y = h - margin + (ball.y - (h - margin)) * 0.3;

      // Update trail
      ball.trail.push({ x: ball.x, y: ball.y });
      if (ball.trail.length > TRAIL_LENGTH) ball.trail.shift();

      // --- Draw ---
      ctx.fillStyle = "#141416";
      ctx.fillRect(0, 0, w, h);

      // Signal strength for glow intensity
      const strength = d
        ? Math.min(
            1,
            Math.sqrt(d.asymmetry * d.asymmetry + (d.concentration - 0.5) ** 2) * 2,
          )
        : 0;

      // Trail
      for (let i = 0; i < ball.trail.length; i++) {
        const t = ball.trail[i];
        const alpha = (i / ball.trail.length) * 0.3;
        const radius = BALL_RADIUS * (0.3 + 0.7 * (i / ball.trail.length));
        ctx.beginPath();
        ctx.arc(t.x, t.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = GLOW_COLOR + alpha + ")";
        ctx.fill();
      }

      // Glow
      const glowRadius = BALL_RADIUS * (2 + strength * 2);
      const gradient = ctx.createRadialGradient(
        ball.x,
        ball.y,
        BALL_RADIUS * 0.5,
        ball.x,
        ball.y,
        glowRadius,
      );
      gradient.addColorStop(0, GLOW_COLOR + (0.3 + strength * 0.3) + ")");
      gradient.addColorStop(1, GLOW_COLOR + "0)");
      ctx.beginPath();
      ctx.arc(ball.x, ball.y, glowRadius, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      // Ball
      ctx.beginPath();
      ctx.arc(ball.x, ball.y, BALL_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = BALL_COLOR;
      ctx.fill();

      // Calibrating label
      if (!d || !d.calibrated) {
        ctx.font = "13px monospace";
        ctx.fillStyle = "#666";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("Calibrating...", w / 2, h / 2 + BALL_RADIUS + 28);
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [initBall]);

  // Reset ball position on resize
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver(() => {
      ballRef.current = null; // re-initialize on next frame
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

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
        minHeight: 0,
      }}
    >
      {/* Canvas */}
      <div ref={containerRef} style={{ flex: 1, minHeight: 0 }}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
      </div>

      {/* Debug readout */}
      <SignalReadout data={data} />
    </div>
  );
}

function SignalReadout({ data }: { data: ControlSignalResponse | null }) {
  if (!data) return null;

  return (
    <div
      style={{
        display: "flex",
        gap: "2rem",
        padding: "0.5rem 1rem",
        fontFamily: "monospace",
        fontSize: "0.75rem",
        color: "#888",
        borderTop: "1px solid #222",
        flexWrap: "wrap",
      }}
    >
      <span>
        Asymmetry (L/R):{" "}
        <span style={{ color: "#aaa" }}>{data.asymmetry.toFixed(3)}</span>
      </span>
      <span>
        Concentration:{" "}
        <span style={{ color: "#aaa" }}>{data.concentration.toFixed(3)}</span>
      </span>
      <span>
        Raw beta/alpha:{" "}
        <span style={{ color: "#aaa" }}>{data.raw_concentration.toFixed(2)}</span>
      </span>
      {!data.calibrated && (
        <span style={{ color: "#666" }}>
          warming up ({data.update_count}/10)
        </span>
      )}
    </div>
  );
}
