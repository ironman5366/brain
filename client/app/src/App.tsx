import { useEEGStream } from "./hooks/useEEGStream";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { EEGDisplay } from "./components/EEGDisplay";

const WS_URL = "ws://localhost:8765/ws/eeg";

function App() {
  const { state, bufferRef } = useEEGStream(WS_URL);

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100vh",
        backgroundColor: "#141416",
        color: "#eee",
      }}
    >
      <ConnectionStatus state={state} />

      {state.connected && state.meta ? (
        <EEGDisplay
          bufferRef={bufferRef}
          channelNames={state.meta.channelNames}
          samplingRate={state.meta.samplingRate}
        />
      ) : (
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
          {state.error
            ? `Error: ${state.error}`
            : "Waiting for connection..."}
        </div>
      )}
    </div>
  );
}

export default App;
