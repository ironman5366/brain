import { useEffect, useState } from "react";
import type { ControlSignalResponse } from "./useControlSignal";

const API_BASE = "http://localhost:8765";

export interface BallPoint {
  x: number;
  y: number;
}

export interface BallTarget {
  x: number;
  y: number;
}

export interface BallStatus {
  state: "idle" | "running";
  sessionId: string | null;
  startedAt: number | null;
  message: string | null;
  ball: {
    x: number;
    y: number;
    vx: number;
    vy: number;
    trail: BallPoint[];
  };
  target: BallTarget | null;
  control: ControlSignalResponse | null;
  tickHz: number;
  windowSec: number;
  connectedClients: number;
}

const INITIAL_STATE: BallStatus = {
  state: "idle",
  sessionId: null,
  startedAt: null,
  message: null,
  ball: {
    x: 0.5,
    y: 0.5,
    vx: 0,
    vy: 0,
    trail: [{ x: 0.5, y: 0.5 }],
  },
  target: null,
  control: null,
  tickHz: 20,
  windowSec: 1,
  connectedClients: 0,
};

export function useBall() {
  const [state, setState] = useState<BallStatus>(INITIAL_STATE);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const es = new EventSource(`${API_BASE}/api/ball/events`);

    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data) as Record<string, unknown>;
        const type = data.type as string;

        if (type === "message") {
          setState((prev) => ({
            ...prev,
            message: (data.text as string) ?? null,
          }));
          setError(null);
          return;
        }

        if (type === "target") {
          setState((prev) => ({
            ...prev,
            target: (data.target as BallTarget | null) ?? null,
          }));
          setError(null);
          return;
        }

        if (type === "stopped") {
          setState((prev) => ({
            ...prev,
            state: "idle",
            sessionId: null,
            startedAt: null,
            ball: INITIAL_STATE.ball,
            control: null,
            message: null,
          }));
          setError(null);
          return;
        }

        setState({
          state: ((data.state as "idle" | "running") ?? "idle"),
          sessionId: (data.session_id as string) ?? null,
          startedAt: (data.started_at as number) ?? null,
          message: (data.message as string) ?? null,
          ball: ((data.ball as BallStatus["ball"]) ?? INITIAL_STATE.ball),
          target: (data.target as BallTarget | null) ?? null,
          control: (data.control as ControlSignalResponse) ?? null,
          tickHz: (data.tick_hz as number) ?? INITIAL_STATE.tickHz,
          windowSec: (data.window_sec as number) ?? INITIAL_STATE.windowSec,
          connectedClients:
            (data.connected_clients as number) ?? INITIAL_STATE.connectedClients,
        });
        setError(null);
      } catch {
        setError("Bad ball event payload");
      }
    };

    es.onerror = () => {
      setError("Ball event stream disconnected");
    };

    return () => {
      es.close();
    };
  }, []);

  return { state, error };
}
