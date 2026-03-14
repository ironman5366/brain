import { useEffect, useRef } from "react";
import { useBall } from "../../hooks/useBall";
import type { ControlSignalResponse } from "../../hooks/useControlSignal";

const BALL_RADIUS = 14;
const GLOW_COLOR = "rgba(124, 111, 224, ";
const BALL_COLOR = "#9d93e8";

export function BallApp() {
  const { state, error } = useBall();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef(state);
  const rafRef = useRef<number>(0);
  stateRef.current = state;

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const loop = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = "#141416";
      ctx.fillRect(0, 0, w, h);

      const current = stateRef.current;
      const control = current.control;
      const strength = control
        ? Math.min(
            1,
            Math.sqrt(
              control.asymmetry * control.asymmetry +
                (control.concentration - 0.5) ** 2,
            ) * 2,
          )
        : 0;

      // Draw target if set
      if (current.target) {
        const tx = current.target.x * w;
        const ty = current.target.y * h;
        const targetRadius = 20;

        // Pulsing ring
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 400);

        // Outer glow
        const tGrad = ctx.createRadialGradient(tx, ty, targetRadius * 0.5, tx, ty, targetRadius * (1.5 + pulse * 0.5));
        tGrad.addColorStop(0, `rgba(255, 140, 50, ${0.15 + pulse * 0.1})`);
        tGrad.addColorStop(1, "rgba(255, 140, 50, 0)");
        ctx.beginPath();
        ctx.arc(tx, ty, targetRadius * (1.5 + pulse * 0.5), 0, Math.PI * 2);
        ctx.fillStyle = tGrad;
        ctx.fill();

        // Ring
        ctx.beginPath();
        ctx.arc(tx, ty, targetRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(255, 140, 50, ${0.6 + pulse * 0.3})`;
        ctx.lineWidth = 2;
        ctx.stroke();

        // Crosshair
        ctx.beginPath();
        ctx.moveTo(tx - targetRadius * 0.4, ty);
        ctx.lineTo(tx + targetRadius * 0.4, ty);
        ctx.moveTo(tx, ty - targetRadius * 0.4);
        ctx.lineTo(tx, ty + targetRadius * 0.4);
        ctx.strokeStyle = `rgba(255, 140, 50, ${0.5 + pulse * 0.3})`;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      for (let i = 0; i < current.ball.trail.length; i++) {
        const point = current.ball.trail[i];
        const alpha = (i / Math.max(current.ball.trail.length, 1)) * 0.3;
        const radius = BALL_RADIUS * (0.3 + 0.7 * (i / Math.max(current.ball.trail.length, 1)));
        ctx.beginPath();
        ctx.arc(point.x * w, point.y * h, radius, 0, Math.PI * 2);
        ctx.fillStyle = GLOW_COLOR + alpha + ")";
        ctx.fill();
      }

      const ballX = current.ball.x * w;
      const ballY = current.ball.y * h;
      const glowRadius = BALL_RADIUS * (2 + strength * 2);
      const gradient = ctx.createRadialGradient(
        ballX,
        ballY,
        BALL_RADIUS * 0.5,
        ballX,
        ballY,
        glowRadius,
      );
      gradient.addColorStop(0, GLOW_COLOR + (0.3 + strength * 0.3) + ")");
      gradient.addColorStop(1, GLOW_COLOR + "0)");
      ctx.beginPath();
      ctx.arc(ballX, ballY, glowRadius, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(ballX, ballY, BALL_RADIUS, 0, Math.PI * 2);
      ctx.fillStyle = BALL_COLOR;
      ctx.fill();

      if (current.state === "idle") {
        drawCenteredLabel(ctx, w, h, "Waiting for an agent to start...");
      } else if (!control || !control.calibrated) {
        drawCenteredLabel(ctx, w, h, "Warming up control signal...");
      }

      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
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
        backgroundColor: "#141416",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "0.5rem 1rem",
          borderBottom: "1px solid #333",
          fontFamily: "monospace",
          fontSize: "0.8rem",
          color: "#888",
          gap: "1rem",
          flexWrap: "wrap",
        }}
      >
        <span>
          Brain Ball
          {state.sessionId && <span style={{ color: "#555" }}> &middot; {state.sessionId}</span>}
        </span>
        <span style={{ color: state.state === "running" ? "#7c6fe0" : "#666" }}>
          {state.state === "running" ? "running" : "idle"}
        </span>
      </div>

      {state.message && (
        <div
          style={{
            padding: "0.5rem 1rem",
            borderBottom: "1px solid #222",
            fontFamily: "monospace",
            fontSize: "0.8rem",
            color: "#ddd",
            backgroundColor: "#1a1a2e",
            lineHeight: 1.5,
          }}
        >
          {state.message}
        </div>
      )}

      <div ref={containerRef} style={{ flex: 1, minHeight: 0 }}>
        <canvas
          ref={canvasRef}
          style={{ width: "100%", height: "100%", display: "block" }}
        />
      </div>

      <SignalReadout
        state={state.state}
        control={state.control}
        connectedClients={state.connectedClients}
        tickHz={state.tickHz}
      />
    </div>
  );
}

function drawCenteredLabel(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  text: string,
) {
  ctx.font = "13px monospace";
  ctx.fillStyle = "#666";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, width / 2, height / 2 + BALL_RADIUS + 28);
}

function SignalReadout({
  state,
  control,
  connectedClients,
  tickHz,
}: {
  state: "idle" | "running";
  control: ControlSignalResponse | null;
  connectedClients: number;
  tickHz: number;
}) {
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
        State: <span style={{ color: "#aaa" }}>{state}</span>
      </span>
      <span>
        Tick Hz: <span style={{ color: "#aaa" }}>{tickHz.toFixed(0)}</span>
      </span>
      <span>
        Viewers: <span style={{ color: "#aaa" }}>{connectedClients}</span>
      </span>
      <span>
        Asymmetry:{" "}
        <span style={{ color: "#aaa" }}>
          {control ? control.asymmetry.toFixed(3) : "n/a"}
        </span>
      </span>
      <span>
        Concentration:{" "}
        <span style={{ color: "#aaa" }}>
          {control ? control.concentration.toFixed(3) : "n/a"}
        </span>
      </span>
      <span>
        Raw beta/alpha:{" "}
        <span style={{ color: "#aaa" }}>
          {control ? control.raw_concentration.toFixed(2) : "n/a"}
        </span>
      </span>
      {control && !control.calibrated && (
        <span style={{ color: "#666" }}>
          warming up ({control.update_count}/10)
        </span>
      )}
    </div>
  );
}
