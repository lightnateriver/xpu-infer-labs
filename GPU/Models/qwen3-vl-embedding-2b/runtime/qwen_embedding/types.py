"""Feature result shared with downstream consumers."""
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class Extraction:
    arrays: dict[str, np.ndarray]
    scores: dict[str, float | int | None]
    extra: dict[str, Any] = field(default_factory=dict)
