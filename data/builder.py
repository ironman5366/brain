# Builtin imports
from typing import Literal
from pathlib import Path

# Internal imports
from data.alljoined import fetch_alljoined
from constants import DEFAULT_WINDOW_SECONDS, DEFAULT_SFREQ, DEFAULT_NORMALIZATION

# External imports
import polars as pl
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    target_sfreq: float = DEFAULT_SFREQ
    normalization: Literal["recording"] | Literal["window"] | Literal["none"] = (
        DEFAULT_NORMALIZATION
    )


def build_dataset():
    alljoined_data = pl.
