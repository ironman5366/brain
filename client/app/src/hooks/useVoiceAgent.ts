import { useState, useEffect, useRef, useCallback } from "react";

const API_BASE = "http://localhost:8765";

/** Seconds of silence before auto-stopping recording */
const SILENCE_TIMEOUT = 1.5;
/** RMS threshold below which audio is considered silence */
const SILENCE_THRESHOLD = 0.01;

export type VoicePhase =
  | "idle"
  | "recording"
  | "transcribing"
  | "playing"
  | "error";

export interface VoiceAgentState {
  phase: VoicePhase;
  question: string | null;
  transcript: string;
  error: string | null;
  /** Status text pushed by the lead agent via voice_notify() */
  statusText: string | null;
}

const INITIAL_STATE: VoiceAgentState = {
  phase: "idle",
  question: null,
  transcript: "",
  error: null,
  statusText: null,
};

export function useVoiceAgent() {
  const [state, setState] = useState<VoiceAgentState>(INITIAL_STATE);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const silenceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const rafRef = useRef<number | null>(null);
  const requestIdRef = useRef<string | null>(null);

  const cleanup = useCallback(() => {
    if (silenceTimerRef.current) {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = null;
    }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state !== "inactive"
    ) {
      mediaRecorderRef.current.stop();
    }
    mediaRecorderRef.current = null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.removeAttribute("src");
      audioRef.current.load();
      audioRef.current.remove();
      audioRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(() => {});
      audioCtxRef.current = null;
    }
    chunksRef.current = [];
  }, []);

  /** Start recording with silence detection */
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up silence detection via AnalyserNode
      const audioCtx = new AudioContext();
      audioCtxRef.current = audioCtx;
      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 2048;
      source.connect(analyser);
      const dataArray = new Float32Array(analyser.fftSize);

      let lastSoundTime = performance.now();
      let hasSpeech = false;

      const checkSilence = () => {
        analyser.getFloatTimeDomainData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i] * dataArray[i];
        }
        const rms = Math.sqrt(sum / dataArray.length);

        if (rms > SILENCE_THRESHOLD) {
          lastSoundTime = performance.now();
          hasSpeech = true;
        }

        // Only auto-stop after we've heard some speech + silence gap
        if (
          hasSpeech &&
          performance.now() - lastSoundTime > SILENCE_TIMEOUT * 1000
        ) {
          stopRecording();
          return;
        }

        rafRef.current = requestAnimationFrame(checkSilence);
      };
      rafRef.current = requestAnimationFrame(checkSilence);

      // Set up MediaRecorder
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      recorder.onstop = async () => {
        // Stop silence detection
        if (rafRef.current) {
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;
        }
        audioCtx.close().catch(() => {});
        audioCtxRef.current = null;

        const blob = new Blob(chunksRef.current, { type: recorder.mimeType });
        if (blob.size === 0) {
          setState(INITIAL_STATE);
          return;
        }
        setState((s) => ({ ...s, phase: "transcribing" }));

        try {
          const form = new FormData();
          const ext = recorder.mimeType.includes("webm") ? "webm" : "ogg";
          form.append("audio", blob, `recording.${ext}`);
          if (requestIdRef.current) {
            form.append("request_id", requestIdRef.current);
          }

          const resp = await fetch(`${API_BASE}/api/voice/transcribe`, {
            method: "POST",
            body: form,
          });
          const data = await resp.json();
          setState((s) => ({ ...s, transcript: data.text || "" }));
        } catch (err) {
          console.error("[voice] transcribe error:", err);
        }

        // Clean up mic
        streamRef.current?.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        requestIdRef.current = null;
        // Brief flash of transcript, then dismiss
        setTimeout(() => setState(INITIAL_STATE), 1500);
      };

      recorder.start();
      setState((s) => ({ ...s, phase: "recording" }));
    } catch {
      setState({
        ...INITIAL_STATE,
        phase: "error",
        error: "Microphone access denied",
      });
      setTimeout(() => setState(INITIAL_STATE), 3000);
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === "recording"
    ) {
      mediaRecorderRef.current.stop();
    }
  }, []);

  /** Handle agent-initiated voice_ask: play TTS then record response */
  const handleVoiceAsk = useCallback(
    (requestId: string, question: string) => {
      requestIdRef.current = requestId;
      setState({ phase: "playing", question, transcript: "", error: null });

      const audio = document.createElement("audio");
      audio.style.display = "none";
      document.body.appendChild(audio);
      audioRef.current = audio;

      audio.onended = () => {
        audio.remove();
        audioRef.current = null;
        startRecording();
      };

      audio.onerror = () => {
        audio.remove();
        audioRef.current = null;
        setState({
          ...INITIAL_STATE,
          phase: "error",
          error: "TTS playback failed",
        });
        setTimeout(() => setState(INITIAL_STATE), 3000);
      };

      const encodedText = encodeURIComponent(question);
      audio.src = `${API_BASE}/api/voice/speak?text=${encodedText}`;
      audio.play().catch(() => {});
    },
    [startRecording],
  );

  // SSE subscription
  useEffect(() => {
    const es = new EventSource(`${API_BASE}/api/voice/events`);

    es.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type === "voice_ask") {
        handleVoiceAsk(data.request_id, data.question);
      } else if (data.type === "status_update") {
        setState((s) => ({ ...s, statusText: data.text }));
      }
    };

    es.onerror = () => {};

    return () => {
      es.close();
      cleanup();
    };
  }, [handleVoiceAsk, cleanup]);

  // Spacebar: press once to start, auto-stops on silence
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== "Space" || e.repeat) return;
      const tag = (e.target as HTMLElement).tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (state.phase !== "idle") return;
      e.preventDefault();
      requestIdRef.current = null;
      startRecording();
    };

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [state.phase, startRecording]);

  return { state };
}
