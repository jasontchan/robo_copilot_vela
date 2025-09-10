import random
from collections import deque
from typing import Callable, Dict, List, Tuple

import numpy as np

# Base type for a single corruptor operation
CorruptOp = Tuple[Callable[[np.ndarray], np.ndarray], float]  # (op_fn, probability)


class SignalCorruptor:
    def __init__(self):
        self.ops: List[CorruptOp] = []

    def register(self, op_fn: Callable[[np.ndarray], np.ndarray], prob: float = 1.0):
        """Add a corruptor operation with probability `prob` (0.0–1.0)."""
        self.ops.append((op_fn, prob))

    def corrupt(self, signal: np.ndarray) -> np.ndarray:
        """Apply each registered operation in sequence (in place) with its probability."""
        out = signal.copy()
        for op_fn, prob in self.ops:
            if random.random() < prob:
                out = op_fn(out)
        return out


def make_delay_op(channels: List[int], delay_steps: int):
    """Delays each specified channel by `delay_steps` via a ring buffer."""
    buffers = {ch: deque(maxlen=delay_steps + 1) for ch in channels}

    def op(signal: np.ndarray) -> np.ndarray:
        out = signal.copy()
        for ch in channels:
            buffers[ch].append(signal[ch])
            # once buffer fills, oldest element is the delayed value
            if len(buffers[ch]) == delay_steps + 1:
                out[ch] = buffers[ch][0]
        return out

    return op


def make_random_scale_op(channels: List[int], scale_range: Tuple[float, float]):
    """Scales each listed channel by a random factor in [min, max]."""

    def op(signal: np.ndarray) -> np.ndarray:
        out = signal.copy()
        for ch in channels:
            factor = random.uniform(*scale_range)
            out[ch] *= factor
        return out

    return op


def make_random_zero_op(channels: List[int]):
    """Zeroes out *one* randomly chosen channel from the list."""

    def op(signal: np.ndarray) -> np.ndarray:
        out = signal.copy()
        ch = random.choice(channels)
        out[ch] = 0.0
        return out

    return op


def make_random_swap_op(channels: List[int]):
    """Swaps the values of two *distinct* randomly chosen channels from the list."""

    def op(signal: np.ndarray) -> np.ndarray:
        out = signal.copy()
        i, j = random.sample(channels, 2)
        out[i], out[j] = out[j], out[i]
        return out

    return op


def make_random_negate_op(channels: List[int], p_channel: float = 0.5):
    """
    Negates each channel in `channels` independently with probability p_channel.
    """

    def op(signal: np.ndarray) -> np.ndarray:
        out = signal.copy()
        for ch in channels:
            if random.random() < p_channel:
                out[ch] = -out[ch]
        return out

    return op


Corruptions: Dict[str, Callable] = {
    "delay": make_delay_op,
    "random_scale": make_random_scale_op,
    "random_zero": make_random_zero_op,
    "random_swap": make_random_swap_op,
    "random_negate": make_random_negate_op,
}
