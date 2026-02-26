import type { StreamState } from "../hooks/useEEGStream";

interface Props {
  state: StreamState;
}

export function ConnectionStatus({ state }: Props) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "1.5rem",
        padding: "0.75rem 1rem",
        borderBottom: "1px solid #333",
        fontFamily: "monospace",
        fontSize: "0.85rem",
      }}
    >
      <StatusDot
        active={state.connected}
        label={state.connected ? "Connected" : "Disconnected"}
      />

      {state.meta && (
        <>
          <span style={{ color: "#aaa" }}>
            {state.meta.samplingRate} Hz
          </span>
          <span style={{ color: "#aaa" }}>
            {state.meta.channelNames.length}ch: {state.meta.channelNames.join(", ")}
          </span>
        </>
      )}

      {state.error && (
        <span style={{ color: "#f44" }}>{state.error}</span>
      )}
    </div>
  );
}

function StatusDot({ active, label }: { active: boolean; label: string }) {
  return (
    <span style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          backgroundColor: active ? "#4caf50" : "#666",
          display: "inline-block",
        }}
      />
      <span style={{ color: active ? "#4caf50" : "#888" }}>{label}</span>
    </span>
  );
}
