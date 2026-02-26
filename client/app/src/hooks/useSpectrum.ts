import { useState, useEffect, useRef } from "react";
import type { SpectrumResponse } from "../lib/spectrum";

const API_BASE = "http://localhost:8765";
const POLL_INTERVAL_MS = 500;

export function useSpectrum(enabled: boolean = true) {
  const [data, setData] = useState<SpectrumResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!enabled) {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }

    const fetchSpectrum = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/spectrum`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json: SpectrumResponse = await resp.json();
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

    fetchSpectrum();
    timerRef.current = setInterval(fetchSpectrum, POLL_INTERVAL_MS);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [enabled]);

  return { data, error };
}
