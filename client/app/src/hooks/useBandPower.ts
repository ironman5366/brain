import { useState, useEffect, useRef } from "react";
import type { BandPowerResponse, BandPowerSnapshot } from "../lib/bandpower";
import { BAND_NAMES } from "../lib/bandpower";

const API_BASE = "http://localhost:8765";
const POLL_INTERVAL_MS = 500;
const MAX_HISTORY = 120; // 60 seconds at 2Hz

export function useBandPower(enabled: boolean = true) {
  const [data, setData] = useState<BandPowerResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<BandPowerSnapshot[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!enabled) {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      return;
    }

    const fetchBandPower = async () => {
      try {
        const resp = await fetch(`${API_BASE}/api/bandpower`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const json: BandPowerResponse = await resp.json();
        if (json.error) {
          setError(json.error);
        } else {
          setData(json);
          setError(null);

          const bandMap = new Map(json.bands.map((b) => [b.name, b.relative]));
          const snapshot: BandPowerSnapshot = {
            timestamp: Date.now(),
            relatives: BAND_NAMES.map((name) => bandMap.get(name) ?? 0),
          };
          setHistory((prev) => {
            const next = [...prev, snapshot];
            return next.length > MAX_HISTORY ? next.slice(-MAX_HISTORY) : next;
          });
        }
      } catch (e) {
        setError((e as Error).message);
      }
    };

    fetchBandPower();
    timerRef.current = setInterval(fetchBandPower, POLL_INTERVAL_MS);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [enabled]);

  return { data, history, error };
}
