"""OpenBCI Cyton 8-channel with Ultracortex (dry electrodes)."""

from brainflow.board_shim import BoardIds, BrainFlowInputParams

from .headset import Headset

# Standard 10-20 positions for Cyton 8-channel Ultracortex Mark IV
CYTON_CHANNEL_NAMES = ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"]

# Physical wire colors for the Ultracortex Mark IV electrode harness
CYTON_WIRE_COLORS: dict[str, str] = {
    "Fp1": "grey",
    "Fp2": "purple",
    "C3": "blue",
    "C4": "green",
    "P7": "yellow",
    "P8": "orange",
    "O1": "red",
    "O2": "brown",
}

# Cyton board pin labels (bottom row N*P pins)
CYTON_PIN_LABELS: dict[str, str] = {
    "Fp1": "N1P",
    "Fp2": "N2P",
    "C3": "N3P",
    "C4": "N4P",
    "P7": "N5P",
    "P8": "N6P",
    "O1": "N7P",
    "O2": "N8P",
}


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

    @property
    def wire_colors(self) -> dict[str, str]:
        return CYTON_WIRE_COLORS

    @property
    def pin_labels(self) -> dict[str, str]:
        return CYTON_PIN_LABELS
