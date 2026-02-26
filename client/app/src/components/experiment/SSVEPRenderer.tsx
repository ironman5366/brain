import { useEffect, useRef } from "react";
import type { SSVEPFrequency } from "../../lib/experiment.types";

interface Props {
  frequencies: SSVEPFrequency[];
  targetFrequencyHz?: number;
}

export function SSVEPRenderer({ frequencies }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;

    // Size canvas to container
    const resize = () => {
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      ctx.scale(dpr, dpr);
    };
    resize();

    startTimeRef.current = performance.now();

    const animate = () => {
      const rect = container.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const elapsed = performance.now() - startTimeRef.current;

      // Clear to black
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      const dpr = window.devicePixelRatio || 1;
      ctx.scale(dpr, dpr);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, w, h);

      for (const freq of frequencies) {
        const halfPeriodMs = 1000 / (2 * freq.hz);
        const phase = Math.floor(elapsed / halfPeriodMs);
        const visible = phase % 2 === 0;

        if (visible) {
          const pos = resolvePosition(freq.position, w, h);
          const size = getSize(freq);
          const color = getColor(freq);

          ctx.fillStyle = color;
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, size / 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafRef.current);
    };
  }, [frequencies]);

  return (
    <div
      ref={containerRef}
      style={{ flex: 1, backgroundColor: "#000", position: "relative" }}
    >
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", top: 0, left: 0 }}
      />
    </div>
  );
}

function resolvePosition(
  position: SSVEPFrequency["position"],
  w: number,
  h: number
): { x: number; y: number } {
  if (typeof position === "object") {
    return position;
  }
  switch (position) {
    case "left":
      return { x: w * 0.25, y: h * 0.5 };
    case "right":
      return { x: w * 0.75, y: h * 0.5 };
    case "center":
    default:
      return { x: w * 0.5, y: h * 0.5 };
  }
}

function getSize(freq: SSVEPFrequency): number {
  if (freq.stimulus.type === "shape") return freq.stimulus.size;
  return 200; // default
}

function getColor(freq: SSVEPFrequency): string {
  if (freq.stimulus.type === "shape") return freq.stimulus.color ?? "#ffffff";
  return "#ffffff";
}
