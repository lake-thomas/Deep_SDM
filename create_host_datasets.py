#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Compatibility CLI for the refactored Host-NAIP-SDM dataset builder.

The implementation now lives in ../deep_sdm/host_dataset_builder
"""

from __future__ import annotations

import sys
from pathlib import Path

_DIR = Path(__file__).resolve().parents[1] / "Host_Datasets"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from host_dataset_builder.pipeline import main


if __name__ == "__main__":
    main()

# EOF
