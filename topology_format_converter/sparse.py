from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np

from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata
from .volume import as_3d_array, to_numpy


@dataclass
class SparseVoxels:
    """Sparse COO voxel representation with integer ijk indices and values.

    ``values`` is ``[N]`` for one value per voxel or ``[N, C]`` for a feature
    vector per voxel, optionally named by ``channel_names``.
    """

    indices: np.ndarray
    values: np.ndarray
    shape: tuple[int, int, int]
    metadata: Optional[ConversionMetadata] = None
    channel_names: Optional[tuple[str, ...]] = None


def _validate_shape(shape: Iterable[int]) -> tuple[int, int, int]:
    values = tuple(int(value) for value in shape)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"shape must contain three positive integers; got {values}.")
    return values


def _validate_indices(indices: Any, shape: tuple[int, int, int]) -> np.ndarray:
    indices_array = np.asarray(indices, dtype=np.int64)
    if indices_array.ndim != 2 or indices_array.shape[1] != 3:
        raise ValueError(f"indices must have shape [N, 3]; got {indices_array.shape}.")
    if len(indices_array) and (
        np.any(indices_array < 0)
        or np.any(indices_array >= np.asarray(shape, dtype=np.int64))
    ):
        raise ValueError(
            "indices contain entries outside the declared sparse voxel shape."
        )
    return indices_array


def _validate_values(values: Any, count: int) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float32)
    if values_array.ndim == 2 and values_array.shape[0] == count:
        return values_array
    values_array = values_array.reshape(-1)
    if values_array.shape != (count,):
        raise ValueError(
            f"values must have shape ({count},) or ({count}, C); got {values_array.shape}."
        )
    return values_array


def _validate_channel_names(names: Optional[Sequence[str]], values: np.ndarray) -> Optional[tuple[str, ...]]:
    if names is None:
        return None
    names = tuple(str(name) for name in names)
    count = values.shape[1] if values.ndim == 2 else 1
    if len(names) != count:
        raise ValueError(f"Expected {count} channel names; got {len(names)}.")
    return names


def make_sparse_voxels(
    indices: Any,
    values: Optional[Any] = None,
    *,
    shape: Iterable[int],
    metadata: Optional[ConversionMetadata] = None,
    channel_names: Optional[Sequence[str]] = None,
) -> SparseVoxels:
    shape_values = _validate_shape(shape)
    indices_array = _validate_indices(indices, shape_values)
    if values is None:
        values_array = np.ones((len(indices_array),), dtype=np.float32)
    else:
        values_array = _validate_values(values, len(indices_array))
    return SparseVoxels(
        indices=indices_array,
        values=values_array,
        shape=shape_values,
        metadata=metadata,
        channel_names=_validate_channel_names(channel_names, values_array),
    )


def dense_to_sparse(volume: Any, *, threshold: Optional[float] = None) -> SparseVoxels:
    array = as_3d_array(volume, name="volume")
    if threshold is None:
        mask = array != 0
    else:
        mask = array >= float(threshold)
    indices = np.argwhere(mask).astype(np.int64, copy=False)
    values = (
        array[tuple(indices.T)].astype(np.float32, copy=False)
        if len(indices)
        else np.empty((0,), dtype=np.float32)
    )
    return make_sparse_voxels(indices, values, shape=array.shape)


def dense_channels_to_sparse(
    channels: Any,
    *,
    mask: Optional[Any] = None,
    channel_names: Optional[Sequence[str]] = None,
) -> SparseVoxels:
    """Sparse voxels with a feature vector per active site from a ``[C, D, H, W]`` grid.

    Active sites are the voxels inside ``mask``, or every voxel with a non-zero
    value in any channel when no mask is given.
    """

    grid = np.asarray(to_numpy(channels), dtype=np.float32)
    if grid.ndim == 3:
        grid = grid[None]
    if grid.ndim != 4:
        raise ValueError(f"channels must have shape [C, D, H, W]; got {grid.shape}.")
    if mask is None:
        active = np.any(grid != 0, axis=0)
    else:
        active = np.asarray(to_numpy(mask)).squeeze().astype(bool)
        if active.shape != grid.shape[1:]:
            raise ValueError(f"mask shape {active.shape} does not match grid shape {grid.shape[1:]}.")
    indices = np.argwhere(active).astype(np.int64, copy=False)
    values = grid[:, indices[:, 0], indices[:, 1], indices[:, 2]].T
    return make_sparse_voxels(indices, values, shape=grid.shape[1:], channel_names=channel_names)


def sparse_to_dense(sparse: SparseVoxels, *, fill_value: float = 0.0) -> np.ndarray:
    """Dense ``[D, H, W]`` grid, or ``[C, D, H, W]`` for multi-channel sparse voxels."""

    sparse = make_sparse_voxels(sparse.indices, sparse.values, shape=sparse.shape)
    if sparse.values.ndim == 2:
        dense = np.full((sparse.values.shape[1], *sparse.shape), float(fill_value), dtype=np.float32)
        if len(sparse.indices):
            dense[(slice(None), *sparse.indices.T)] = sparse.values.T
        return dense
    dense = np.full(sparse.shape, float(fill_value), dtype=np.float32)
    if len(sparse.indices):
        dense[tuple(sparse.indices.T)] = sparse.values
    return dense


def load_sparse_voxels(
    path: str | Path,
    *,
    indices_key: str = "indices",
    values_key: str = "values",
    shape_key: str = "shape",
) -> SparseVoxels:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix != ".npz":
        raise ValueError(f"Unsupported sparse voxel extension {suffix!r}. Use .npz.")
    with np.load(path, allow_pickle=False) as data:
        index_key = indices_key
        if index_key not in data:
            for candidate in ("indices", "coords", "ijk"):
                if candidate in data:
                    index_key = candidate
                    break
        value_key = values_key
        if value_key not in data:
            for candidate in ("values", "value", "occupancy", "density"):
                if candidate in data:
                    value_key = candidate
                    break
        actual_shape_key = shape_key
        if actual_shape_key not in data:
            for candidate in ("shape", "grid_shape", "volume_shape"):
                if candidate in data:
                    actual_shape_key = candidate
                    break
        if index_key not in data:
            raise KeyError(
                f"Could not find sparse voxel indices in {path}. Available keys: {data.files}"
            )
        if actual_shape_key not in data:
            raise KeyError(
                f"Could not find sparse voxel shape in {path}. Available keys: {data.files}"
            )
        values = data[value_key] if value_key in data else None
        names = [str(name) for name in data["channel_names"]] if "channel_names" in data else None
        return make_sparse_voxels(data[index_key], values, shape=data[actual_shape_key], channel_names=names)


def save_sparse_voxels(
    sparse: SparseVoxels,
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Sparse voxels can only be saved to .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse = make_sparse_voxels(
        sparse.indices, sparse.values, shape=sparse.shape, channel_names=sparse.channel_names
    )
    arrays: dict[str, Any] = {
        "indices": sparse.indices,
        "values": sparse.values,
        "shape": np.asarray(sparse.shape, dtype=np.int64),
    }
    if sparse.channel_names is not None:
        arrays["channel_names"] = np.asarray(sparse.channel_names)
    np.savez_compressed(path, **arrays)
    sidecar_metadata = metadata or sparse.metadata
    if sidecar_metadata is not None and write_sidecar:
        write_metadata(sidecar_metadata, metadata_sidecar_path(path))
    return path


def sparse_voxels_summary(sparse: SparseVoxels) -> dict[str, Any]:
    sparse = make_sparse_voxels(sparse.indices, sparse.values, shape=sparse.shape)
    return {
        "shape": tuple(int(value) for value in sparse.shape),
        "nnz": int(len(sparse.indices)),
        "value_min": float(np.min(sparse.values)) if len(sparse.values) else 0.0,
        "value_max": float(np.max(sparse.values)) if len(sparse.values) else 0.0,
    }
