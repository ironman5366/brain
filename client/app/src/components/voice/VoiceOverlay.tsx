import type { CSSProperties } from "react";
import { useVoiceAgent, type VoicePhase } from "../../hooks/useVoiceAgent";

const barStyle: CSSProperties = {
  position: "fixed",
  top: 0,
  left: 0,
  right: 0,
  zIndex: 9999,
  backgroundColor: "rgba(20, 20, 30, 0.95)",
  borderBottom: "1px solid #333",
  padding: "0.5rem 1rem",
  fontFamily: "monospace",
  display: "flex",
  alignItems: "center",
  gap: "0.75rem",
  backdropFilter: "blur(8px)",
};

const PHASE_CONFIG: Record<
  Exclude<VoicePhase, "idle">,
  { label: string; color: string }
> = {
  recording: { label: "Recording...", color: "#e04040" },
  transcribing: { label: "Transcribing...", color: "#888" },
  playing: { label: "AI Speaking", color: "#7c6fe0" },
  error: { label: "Error", color: "#e04040" },
};

export function VoiceOverlay() {
  const { state } = useVoiceAgent();

  if (state.phase === "idle") {
    return (
      <div
        style={{
          ...barStyle,
          backgroundColor: "rgba(20, 20, 30, 0.7)",
          justifyContent: state.statusText ? "flex-start" : "center",
        }}
      >
        {state.statusText ? (
          <>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                backgroundColor: "#7c6fe0",
                flexShrink: 0,
                animation: "voicePulse 2s ease-in-out infinite",
              }}
            />
            <span style={{ fontSize: "0.75rem", color: "#999", flex: 1 }}>
              {state.statusText}
            </span>
            <span style={{ fontSize: "0.7rem", color: "#555" }}>
              space to talk
            </span>
          </>
        ) : (
          <span style={{ fontSize: "0.75rem", color: "#666" }}>
            Press space to talk
          </span>
        )}
        <style>{`
          @keyframes voicePulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(1.3); }
          }
        `}</style>
      </div>
    );
  }

  const config = PHASE_CONFIG[state.phase];

  return (
    <>
      <div style={barStyle}>
        <span
          style={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            backgroundColor: config.color,
            flexShrink: 0,
            animation: "voicePulse 1.5s ease-in-out infinite",
          }}
        />
        <span
          style={{
            fontSize: "0.8rem",
            fontWeight: 600,
            color: config.color,
            flexShrink: 0,
          }}
        >
          {config.label}
        </span>
        <span
          style={{
            fontSize: "0.8rem",
            color: "#999",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            flex: 1,
          }}
        >
          {state.transcript || state.question || ""}
        </span>
        {state.error && (
          <span style={{ fontSize: "0.8rem", color: "#e04040", flexShrink: 0 }}>
            {state.error}
          </span>
        )}
      </div>

      <style>{`
        @keyframes voicePulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(1.3); }
        }
      `}</style>
    </>
  );
}
