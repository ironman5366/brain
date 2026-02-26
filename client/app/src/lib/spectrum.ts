export interface SpectrumResponse {
  frequencies: number[];
  amplitudes_db: number[];
  nfft: number;
  sampling_rate: number;
  num_channels: number;
  window_samples: number;
  error?: string;
}
