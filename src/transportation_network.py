"""交通网络定义，对应 MATLAB 中的 transportation_network.m。"""
from __future__ import annotations

import numpy as np
from typing import Dict


def transportation_network() -> Dict[str, np.ndarray]:
    bus = np.array([
        [1, 100, 1],
        [2, 100, 3],
        [3, 100, 6],
        [4, 100, 10],
        [5, 100, 14],
        [6, 100, 24],
    ], dtype=float)

    branch = np.array([
        [1, 2, 1],
        [1, 3, 4],
        [1, 4, 2],
        [2, 4, 3],
        [2, 5, 2],
        [3, 4, 5],
        [4, 5, 3],
        [5, 6, 3],
        [3, 6, 2],
    ], dtype=float)

    initial_status = np.array([0, 0, 0, 0, 1, 0], dtype=float)
    end_status = np.array([1, 0, 0, 0, 0, 0], dtype=float)

    return {
        "bus": bus,
        "branch": branch,
        "initial_status": initial_status,
        "end_status": end_status,
    }
