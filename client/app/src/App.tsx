import { useState } from "react";
import { useEEGStream } from "./hooks/useEEGStream";
import { ConnectionStatus } from "./components/ConnectionStatus";
import { EEGDisplay } from "./components/eeg/EEGDisplay";
import { ImpedanceCheck } from "./components/impedance/ImpedanceCheck";
import { BandPowerApp } from "./components/bandpower/BandPowerApp";
import { FFTApp } from "./components/fft/FFTApp";
import { ExperimentApp } from "./components/experiment/ExperimentApp";
import { BCIApp } from "./components/bci/BCIApp";
import { CalibrationApp } from "./components/calibration/CalibrationApp";
import { Dashboard } from "./components/Dashboard";
import type { AppId } from "./components/Dashboard";

const WS_URL = "ws://localhost:8765/ws/eeg";

type View = "dashboard" | AppId;

function App() {
  const [view, setView] = useState<View>("dashboard");
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
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          borderBottom: "1px solid #333",
          overflow: "hidden",
          flexShrink: 0,
        }}
      >
        {view !== "dashboard" && (
          <button
            onClick={() => setView("dashboard")}
            style={{
              padding: "0.75rem 1rem",
              background: "none",
              border: "none",
              color: "#888",
              fontFamily: "monospace",
              fontSize: "0.85rem",
              cursor: "pointer",
              borderRight: "1px solid #333",
            }}
          >
            &larr; Back
          </button>
        )}
        <ConnectionStatus state={state} />
      </div>

      {/* Content */}
      {view === "dashboard" && (
        <Dashboard onSelectApp={(id) => setView(id)} />
      )}

      {view === "eeg" &&
        (state.connected && state.meta ? (
          <EEGDisplay
            bufferRef={bufferRef}
            channelNames={state.meta.channelNames}
            samplingRate={state.meta.samplingRate}
          />
        ) : (
          <Placeholder
            text={
              state.error
                ? `Error: ${state.error}`
                : "Waiting for connection..."
            }
          />
        ))}

      {view === "impedance" && (
        <ImpedanceCheck
          channelNames={state.meta?.channelNames ?? []}
        />
      )}

      {view === "bandpower" &&
        (state.connected && state.meta ? (
          <BandPowerApp />
        ) : (
          <Placeholder
            text={
              state.error
                ? `Error: ${state.error}`
                : "Waiting for connection..."
            }
          />
        ))}

      {view === "fft" &&
        (state.connected && state.meta ? (
          <FFTApp />
        ) : (
          <Placeholder
            text={
              state.error
                ? `Error: ${state.error}`
                : "Waiting for connection..."
            }
          />
        ))}

      {view === "experiment" && <ExperimentApp />}
      {view === "bci" && <BCIApp />}
      {view === "calibration" && <CalibrationApp />}
    </div>
  );
}

function Placeholder({ text }: { text: string }) {
  return (
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
      {text}
    </div>
  );
}

export default App;
