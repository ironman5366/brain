import { useEffect, useRef } from "react";
import { useBCI } from "../../hooks/useBCI";

const COLS = 6;
const ROWS = 6;

export function BCIApp() {
  const { state, submitFeedback, matrix } = useBCI();

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", backgroundColor: "#141416" }}>
      {/* Header bar */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0.5rem 1rem",
        borderBottom: "1px solid #333",
        fontFamily: "monospace",
        fontSize: "0.8rem",
      }}>
        <span style={{ color: "#888" }}>
          BCI Speller
          {state.sessionId && <span style={{ color: "#555" }}> &middot; {state.sessionId}</span>}
        </span>
        <span style={{ color: modeColor(state.mode) }}>
          {modeLabel(state.mode)}
        </span>
      </div>

      {/* Spelled so far */}
      {state.spelled && (
        <div style={{
          padding: "0.5rem 1rem",
          fontFamily: "monospace",
          fontSize: "1.2rem",
          color: "#7c6fe0",
          borderBottom: "1px solid #222",
          letterSpacing: "0.1em",
        }}>
          {state.spelled}
          <span style={{ opacity: 0.4 }}>_</span>
        </div>
      )}

      {/* Main content area */}
      <div style={{ flex: 1, position: "relative" }}>
        {state.mode === "idle" && <IdleView />}
        {state.mode === "ready" && <ReadyView />}
        {(state.mode === "flashing" || state.mode === "ready") && (
          <MatrixCanvas
            matrix={matrix}
            highlightedRow={state.highlightedRow}
            highlightedCol={state.highlightedCol}
            flashing={state.mode === "flashing"}
            progress={state.flashProgress}
          />
        )}
        {state.mode === "proposing" && (
          <ProposalView
            letter={state.proposedLetter!}
            message={state.proposalMessage}
            onAccept={() => submitFeedback(true)}
            onReject={() => submitFeedback(false)}
          />
        )}
        {state.mode === "message" && (
          <MessageView text={state.messageText!} />
        )}
      </div>
    </div>
  );
}

function IdleView() {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      gap: "1rem",
      fontFamily: "monospace",
    }}>
      <span style={{ fontSize: "1.2rem", color: "#666" }}>
        Waiting for Claude to start...
      </span>
      <span style={{ fontSize: "0.8rem", color: "#444" }}>
        Open the BCI page, then tell Claude you're ready.
      </span>
    </div>
  );
}

function ReadyView() {
  return (
    <div style={{
      position: "absolute",
      top: "1rem",
      left: 0,
      right: 0,
      textAlign: "center",
      fontFamily: "monospace",
      fontSize: "0.9rem",
      color: "#555",
      zIndex: 1,
    }}>
      Think of a letter and focus on it in the matrix.
    </div>
  );
}

function ProposalView({
  letter,
  message,
  onAccept,
  onReject,
}: {
  letter: string;
  message: string | null;
  onAccept: () => void;
  onReject: () => void;
}) {
  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      gap: "1.5rem",
      fontFamily: "monospace",
    }}>
      <span style={{ fontSize: "0.9rem", color: "#888" }}>
        Claude proposes:
      </span>
      <span style={{
        fontSize: "5rem",
        fontWeight: "bold",
        color: "#7c6fe0",
        lineHeight: 1,
      }}>
        {letter}
      </span>
      {message && (
        <span style={{
          fontSize: "0.85rem",
          color: "#888",
          maxWidth: "400px",
          textAlign: "center",
          lineHeight: 1.5,
        }}>
          {message}
        </span>
      )}
      <div style={{ display: "flex", gap: "1rem", marginTop: "0.5rem" }}>
        <button onClick={onAccept} style={buttonStyle("#2d5a2d", "#3a7a3a")}>
          Accept
        </button>
        <button onClick={onReject} style={buttonStyle("#5a2d2d", "#7a3a3a")}>
          Reject
        </button>
      </div>
    </div>
  );
}

function MessageView({ text }: { text: string }) {
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      height: "100%",
      fontFamily: "monospace",
      fontSize: "1rem",
      color: "#aaa",
      padding: "2rem",
      textAlign: "center",
      lineHeight: 1.6,
    }}>
      {text}
    </div>
  );
}

function MatrixCanvas({
  matrix,
  highlightedRow,
  highlightedCol,
  flashing,
  progress,
}: {
  matrix: string[];
  highlightedRow: number | null;
  highlightedCol: number | null;
  flashing: boolean;
  progress: number;
}) {
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
      const cellSize = Math.min((w - 80) / COLS, (h - 80) / ROWS, 90);
      const gridW = cellSize * COLS;
      const gridH = cellSize * ROWS;
      const gridX = (w - gridW) / 2;
      const gridY = (h - gridH) / 2;

      // Draw grid
      const fontSize = Math.round(cellSize * 0.45);
      ctx.font = `bold ${fontSize}px monospace`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";

      for (let row = 0; row < ROWS; row++) {
        for (let col = 0; col < COLS; col++) {
          const x = gridX + col * cellSize;
          const y = gridY + row * cellSize;
          const char = matrix[row * COLS + col];
          const isHighlighted =
            flashing &&
            (row === highlightedRow || col === highlightedCol);

          if (isHighlighted) {
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            ctx.fillStyle = "#000000";
          } else {
            ctx.fillStyle = "#1a1a2e";
            ctx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);
            ctx.fillStyle = "#999";
          }

          ctx.fillText(char, x + cellSize / 2, y + cellSize / 2);
        }
      }

      // Flash progress bar
      if (flashing) {
        ctx.fillStyle = "#222";
        ctx.fillRect(gridX, gridY + gridH + 15, gridW, 4);
        ctx.fillStyle = "#7c6fe0";
        ctx.fillRect(gridX, gridY + gridH + 15, gridW * progress, 4);
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(rafRef.current);
  }, [matrix, highlightedRow, highlightedCol, flashing, progress]);

  return (
    <div
      ref={containerRef}
      style={{ position: "absolute", top: 0, left: 0, right: 0, bottom: 0 }}
    >
      <canvas
        ref={canvasRef}
        style={{ position: "absolute", top: 0, left: 0 }}
      />
    </div>
  );
}

function modeColor(mode: string): string {
  switch (mode) {
    case "idle": return "#555";
    case "ready": return "#4a9";
    case "flashing": return "#e8a838";
    case "proposing": return "#7c6fe0";
    case "message": return "#888";
    default: return "#555";
  }
}

function modeLabel(mode: string): string {
  switch (mode) {
    case "idle": return "Idle";
    case "ready": return "Ready";
    case "flashing": return "Flashing";
    case "proposing": return "Proposal";
    case "message": return "Message";
    default: return mode;
  }
}

function buttonStyle(bg: string, hover: string): React.CSSProperties {
  return {
    padding: "0.75rem 2rem",
    fontFamily: "monospace",
    fontSize: "1rem",
    fontWeight: "bold",
    border: "1px solid #555",
    borderRadius: 8,
    cursor: "pointer",
    color: "#eee",
    backgroundColor: bg,
  };
}
