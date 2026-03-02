"""Base class for headset-specific configuration and behavior."""

from abc import ABC, abstractmethod

from brainflow.board_shim import BrainFlowInputParams


class Headset(ABC):
    """
    Encapsulates everything that varies between EEG headsets.

    To add support for a new headset, create a subclass (e.g. muse.py)
    and implement all abstract members.
    """

    @property
    @abstractmethod
    def board_id(self) -> int:
        """BrainFlow board ID."""

    @property
    @abstractmethod
    def channel_names(self) -> list[str]:
        """10-20 electrode position labels."""

    @abstractmethod
    def get_board_params(self, serial_port: str = "") -> BrainFlowInputParams:
        """Return BrainFlowInputParams configured for this headset."""

    # -- Impedance measurement constants --

    @property
    @abstractmethod
    def lead_off_current_a(self) -> float:
        """Injected lead-off detection current in amps."""

    @property
    @abstractmethod
    def series_resistance_ohm(self) -> float:
        """Series resistance on each electrode input in ohms."""

    @property
    @abstractmethod
    def lead_off_freq_hz(self) -> float:
        """Lead-off detection frequency in Hz."""

    @abstractmethod
    def get_impedance_thresholds(self) -> dict[str, float]:
        """
        Return impedance quality thresholds in ohms.

        Keys: "good" (upper bound for good), "ok" (upper bound for acceptable).
        Anything above "ok" is bad.
        """

    @property
    def wire_colors(self) -> dict[str, str]:
        """Map of channel name → physical wire color. Override per headset."""
        return {}

    @property
    def pin_labels(self) -> dict[str, str]:
        """Map of channel name → board pin label. Override per headset."""
        return {}
