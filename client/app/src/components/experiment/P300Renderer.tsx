import { useEffect, useRef } from "react";

interface Props {
  matrix: string[];
  targetLetter: string;
  highlightedRow: number | null;
  highlightedCol: number | null;
  charPhase: "pre" | "flashing" | "post";
  currentCharIndex: number;
  totalChars: number;
}

const COLS = 6;
const ROWS = 6;

export function P300Renderer({
  matrix,
  targetLetter,
  highlightedRow,
  highlightedCol,
  charPhase,
  currentCharIndex,
  totalChars,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d")!;

    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;

      // Resize canvas if needed
      if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvas.style.width = `${w}px`;
        canvas.style.height = `${h}px`;
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // Clear
      ctx.fillStyle = "#141416";
      ctx.fillRect(0, 0, w, h);

      // Layout
      const cellSize = Math.min((w - 80) / COLS, (h - 120) / ROWS, 90);
      const gridW = cellSize * COLS;
      const gridH = cellSize * ROWS;
      const gridX = (w - gridW) / 2;
      const gridY = (h - gridH) / 2 + 30;

      // Target letter above matrix
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      if (charPhase === "pre") {
        ctx.font = `bold ${Math.round(cellSize * 0.6)}px monospace`;
        ctx.fillStyle = "#7c6fe0";
        ctx.fillText(`Attend to: ${targetLetter}`, w / 2, gridY - 35);
      } else if (charPhase === "flashing") {
        ctx.font = `${Math.round(cellSize * 0.25)}px monospace`;
        ctx.fillStyle = "#444";
        ctx.fillText(`Target: ${targetLetter}`, w / 2, gridY - 35);
      } else {
        ctx.font = `${Math.round(cellSize * 0.3)}px monospace`;
        ctx.fillStyle = "#555";
        ctx.fillText(`Character ${currentCharIndex + 1} of ${totalChars} complete`, w / 2, gridY - 35);
      }

      // Draw grid
      const fontSize = Math.round(cellSize * 0.45);
      ctx.font = `bold ${fontSize}px monospace`;

      for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
          const x = gridX + col * cellSize;
          const y = gridY + row * cellSize;
          const char = matrix[row * COLS + col];
          const isHighlighted =
            charPhase === "flashing" &&
            (row === highlightedRow || col === highlightedCol);

          // Cell background
          if (isHighlighted) {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            ctx.fillStyle = "#000000";
          } else {
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            ctx.fillStyle = "#999";
          }

          // Character
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(char, x + cellSize / 2, y + cellSize / 2);
        }
      }

      // Progress bar at bottom
      ctx.fillStyle = "#222";
      ctx.fillRect(gridX, gridY + gridH + 15, gridW, 4);
      const progress = (currentCharIndex + (charPhase === "post" ? 1 : 0)) / totalChars;
      ctx.fillStyle = "#7c6fe0";
      ctx.fillRect(gridX, gridY + gridH + 15, gridW * progress, 4);

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [matrix, targetLetter, highlightedRow, highlightedCol, charPhase, currentCharIndex, totalChars]);

  return (
    <div
      ref={containerRef}
      style={{ flex: 1, backgroundColor: "#141416", position: "relative" }}
    >
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", top: 0, left: 0 }}
      />
    </div>
  );
}
