import type { RingBuffer } from "../hooks/useEEGStream";
import { EEGCanvas } from "./EEGCanvas";

interface Props {
  bufferRef: React.RefObject<RingBuffer | null>;
  channelNames: string[];
  samplingRate: number;
}

export function EEGDisplay({ bufferRef, channelNames, samplingRate }: Props) {
  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      <EEGCanvas
        bufferRef={bufferRef}
        channelNames={channelNames}
        samplingRate={samplingRate}
      />
    </div>
  );
}
