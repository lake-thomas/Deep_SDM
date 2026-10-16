"""Small dependency-free helpers for reproducible tiled inference."""

from __future__ import annotations

import hashlib
from pathlib import Path


def stable_tile_identity(tile_path: str | Path) -> str:
    """Return a mount-independent state/year/raw/tile identity when available."""
    path = Path(tile_path)
    parts = path.parts
    return "/".join(parts[-5:] if len(parts) >= 5 else parts)


def derive_tile_seed(base_seed: int, tile_path: str | Path) -> int:
    """Derive an order-independent 31-bit RNG seed for one source tile."""
    base_seed = int(base_seed)
    if not 0 <= base_seed <= 2**32 - 1:
        raise ValueError("Inference base seed must be between 0 and 2^32 - 1")
    identity = stable_tile_identity(tile_path)
    digest = hashlib.sha256(f"{base_seed}\0{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)
