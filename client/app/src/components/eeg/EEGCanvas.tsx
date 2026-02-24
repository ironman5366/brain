import { useEffect, useRef } from "react";
import {
  UnifiedLinePlot,
  createWebGL2Context,
  clearCanvas,
  setBackgroundColor,
} from "webgl-plot";
import type { LineConfig } from "webgl-plot";
import type { RingBuffer } from "../../hooks/useEEGStream";

interface Props {
  bufferRef: React.RefObject<RingBuffer | null>;
  channelNames: string[];
  samplingRate: number;
}

const CHANNEL_COLORS: [number, number, number, number][] = [
  [0.35, 0.70, 0.90, 1], // Fp1 - sky blue
  [0.99, 0.55, 0.24, 1], // Fp2 - orange
  [0.30, 0.69, 0.29, 1], // C3  - green
  [0.89, 0.28, 0.20, 1], // C4  - red
  [0.60, 0.40, 0.80, 1], // P7  - purple
  [0.90, 0.67, 0.22, 1], // P8  - gold
  [0.85, 0.37, 0.68, 1], // O1  - pink
  [0.45, 0.77, 0.66, 1], // O2  - teal
];

// How many seconds of data to show on screen
const DISPLAY_SECONDS = 4;

function sizeCanvas(canvas: HTMLCanvasElement) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
}

export function EEGCanvas({ bufferRef, channelNames, samplingRate }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);

  const numChannels = channelNames.length;
  const displaySamples = Math.floor(samplingRate * DISPLAY_SECONDS);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Size the canvas backing store to match its CSS layout size
    sizeCanvas(canvas);

    let gl: WebGL2RenderingContext;
    try {
      gl = createWebGL2Context(canvas);
    } catch (e) {
      console.error("Failed to create WebGL2 context:", e);
      return;
    }
    setBackgroundColor(gl, [0.08, 0.08, 0.10, 1]);

    const plot = new UnifiedLinePlot(gl, numChannels);

    // Create line configs — each channel gets a horizontal band
    const bandHeight = 2.0 / numChannels; // NDC goes from -1 to 1
    const configs: LineConfig[] = [];

    for (let ch = 0; ch < numChannels; ch++) {
      // Interleaved x,y pairs: x goes from -1 to 1, y starts at 0
      const points = new Float32Array(displaySamples * 2);
      for (let i = 0; i < displaySamples; i++) {
        points[i * 2] = (i / (displaySamples - 1)) * 2 - 1; // x: -1 to 1
        points[i * 2 + 1] = 0; // y: 0
      }

      // Stack channels top to bottom
      const yOffset = 1 - bandHeight * (ch + 0.5);

      configs.push({
        points,
        color: CHANNEL_COLORS[ch % CHANNEL_COLORS.length],
        thickness: 1,
        scale: [1, bandHeight * 0.4], // scale Y to fit within band
        offset: [0, yOffset],
      });
    }

    plot.initLines(configs);

    // Handle resize
    const observer = new ResizeObserver(() => {
      sizeCanvas(canvas);
      gl.viewport(0, 0, canvas.width, canvas.height);
    });
    observer.observe(canvas);

    // Animation loop
    const yBuffer = new Float32Array(displaySamples);

    const animate = () => {
      const buf = bufferRef.current;
      if (buf && buf.totalWritten > 0) {
        for (let ch = 0; ch < numChannels && ch < buf.channels.length; ch++) {
          const channelData = buf.channels[ch];

          // Read from ring buffer: oldest visible data to newest
          const startIdx =
            buf.totalWritten >= displaySamples
              ? (buf.writeIndex - displaySamples + buf.capacity) % buf.capacity
              : 0;

          // Track min/max for auto-scaling
          let min = Infinity;
          let max = -Infinity;

          const samplesToRead = Math.min(displaySamples, buf.totalWritten);
          for (let i = 0; i < displaySamples; i++) {
            if (i < samplesToRead) {
              const idx = (startIdx + i) % buf.capacity;
              const val = channelData[idx];
              yBuffer[i] = val;
              if (val < min) min = val;
              if (val > max) max = val;
            } else {
              yBuffer[i] = 0;
            }
          }

          // Normalize to roughly [-1, 1] for display
          const range = max - min;
          if (range > 0) {
            const mid = (max + min) / 2;
            for (let i = 0; i < displaySamples; i++) {
              yBuffer[i] = (yBuffer[i] - mid) / (range * 0.5);
            }
          }

          plot.updateLineY(ch, yBuffer);
        }
      }

      clearCanvas(gl);
      plot.draw();
      animRef.current = requestAnimationFrame(animate);
    };

    animRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animRef.current);
      observer.disconnect();
      plot.cleanup();
    };
  }, [numChannels, samplingRate, displaySamples, bufferRef]);

  return (
    <div ref={containerRef} style={{ position: "relative", flex: 1, minHeight: 0 }}>
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
      {/* Channel labels */}
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 0,
          bottom: 0,
          width: 50,
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-around",
          pointerEvents: "none",
        }}
      >
        {channelNames.map((name, i) => (
          <span
            key={name}
            style={{
              color: `rgba(${CHANNEL_COLORS[i % CHANNEL_COLORS.length]
                .slice(0, 3)
                .map((c) => Math.round(c * 255))
                .join(",")}, 0.9)`,
              fontSize: "0.75rem",
              fontFamily: "monospace",
              fontWeight: "bold",
              paddingLeft: 6,
            }}
          >
            {name}
          </span>
        ))}
      </div>
    </div>
  );
}
