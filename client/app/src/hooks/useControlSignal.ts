import { useState, useEffect, useRef } from "react";

const API_BASE = "http://localhost:8765";
const POLL_INTERVAL_MS = 200;

export interface ControlSignalResponse {
  asymmetry: number;
  concentration: number;
  raw_asymmetry: number;
  raw_concentration: number;
  calibrated: boolean;
  update_count: number;
  per_channel: Record<
    string,
    {
      alpha: number;
      beta: number;
      rejected: boolean;
      issues?: string[];
      rms_uv?: number;
      line_noise_db?: number;
    }
  >;
  error?: string;
}

export function useControlSignal(enabled: boolean = true) {
  const [data, setData] = useState<ControlSignalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!enabled) {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }

    const fetchControl = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/control`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json: ControlSignalResponse = await resp.json();
        if (json.error) {
          setError(json.error);
        } else {
          setData(json);
          setError(null);
        }
      } catch (e) {
        setError((e as Error).message);
      }
    };

    fetchControl();
    timerRef.current = setInterval(fetchControl, POLL_INTERVAL_MS);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [enabled]);

  return { data, error };
}
