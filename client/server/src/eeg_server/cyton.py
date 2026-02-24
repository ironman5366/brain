"""OpenBCI Cyton 8-channel with Ultracortex (dry electrodes)."""

from brainflow.board_shim import BoardIds, BrainFlowInputParams

from .headset import Headset

# Standard 10-20 positions for Cyton 8-channel Ultracortex Mark IV
CYTON_CHANNEL_NAMES = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]


class CytonHeadset(Headset):
    @property
    def board_id(self) -> int:
        return BoardIds.CYTON_BOARD.value

    @property
    def channel_names(self) -> list[str]:
        return CYTON_CHANNEL_NAMES

    def get_board_params(self, serial_port: str = "") -> BrainFlowInputParams:
        params = BrainFlowInputParams()
        if serial_port:
            params.serial_port = serial_port
        return params

    # ADS1299 lead-off detection parameters
    @property
    def lead_off_current_a(self) -> float:
        return 6e-9  # 6 nA

    @property
    def series_resistance_ohm(self) -> float:
        return 2200.0  # 2.2 kOhm

    @property
    def lead_off_freq_hz(self) -> float:
        return 31.2  # Hz

    def get_impedance_thresholds(self) -> dict[str, float]:
        # Dry electrode thresholds (Ultracortex)
        return {
            "good": 50_000.0,   # < 50 kOhm
            "ok": 200_000.0,    # < 200 kOhm
        }
