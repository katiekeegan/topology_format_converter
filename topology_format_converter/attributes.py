from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree

from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata
from .volume import coords_to_voxel_indices, to_numpy, voxel_indices_to_coords


@dataclass
class AttributedPoints:
    """Points carrying a multi-channel feature vector each.

    Unlike :class:`FieldSamples`, which stores one scalar per point, this holds
    any number of named channels per point, for example loads, boundary
    conditions, and design-domain flags of a topology optimization problem.
    """

    points: np.ndarray
    features: np.ndarray
    channel_names: tuple[str, ...]
    metadata: Optional[ConversionMetadata] = None


def as_channel_grid(channels: Any, *, name: str = "channels") -> np.ndarray:
    """Return a channel-first ``[C, D, H, W]`` float32 grid (a 3-D grid becomes one channel)."""

    array = np.asarray(to_numpy(channels), dtype=np.float32)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4:
        raise ValueError(f"{name} must have shape [C, D, H, W] or [D, H, W]; got {array.shape}.")
    return array


def _channel_names(names: Optional[Sequence[str]], count: int) -> tuple[str, ...]:
    if names is None:
        return tuple(f"channel_{index}" for index in range(count))
    names = tuple(str(name) for name in names)
    if len(names) != count:
        raise ValueError(f"Expected {count} channel names; got {len(names)}.")
    return names


def make_attributed_points(
    points: Any,
    features: Any,
    *,
    channel_names: Optional[Sequence[str]] = None,
    metadata: Optional[ConversionMetadata] = None,
) -> AttributedPoints:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3]; got {points_array.shape}.")
    features_array = np.asarray(features, dtype=np.float32)
    if features_array.ndim == 1:
        features_array = features_array[:, None]
    if features_array.ndim != 2 or features_array.shape[0] != len(points_array):
        raise ValueError(
            f"features must have shape [{len(points_array)}, C]; got {features_array.shape}."
        )
    return AttributedPoints(
        points=points_array,
        features=features_array,
        channel_names=_channel_names(channel_names, features_array.shape[1]),
        metadata=metadata,
    )


def sample_grid_channels(
    channels: Any,
    points: Any,
    *,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    """Interpolate every channel of a ``[C, D, H, W]`` grid at points; returns ``[N, C]``.

    ``order=0`` takes the nearest voxel, ``order=1`` interpolates trilinearly.
    Interpolation is right for smooth fields; use :func:`scatter_grid_channels_to_points`
    for point quantities such as loads, which interpolation would smear or miss.
    """

    grid = as_channel_grid(channels)
    indices = coords_to_voxel_indices(points, grid.shape[1:], coordinate_mode=coordinate_mode, spacing=spacing)
    coordinates = [indices[:, 0], indices[:, 1], indices[:, 2]]
    columns = [map_coordinates(channel, coordinates, order=order, mode=mode) for channel in grid]
    return np.stack(columns, axis=1).astype(np.float32, copy=False)


def scatter_grid_channels_to_points(
    channels: Any,
    points: Any,
    *,
    mask: Optional[Any] = None,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Move each voxel's channel values to its nearest point and sum them; returns ``[N, C]``.

    Totals are conserved (the summed loads on the points equal the summed loads
    on the grid), which is what point loads and other extensive quantities need.
    Only voxels with a non-zero value in any channel, or inside ``mask`` when
    given, are moved.
    """

    grid = as_channel_grid(channels)
    points_array = np.asarray(points, dtype=np.float32)
    if mask is None:
        voxel_mask = np.any(grid != 0, axis=0)
    else:
        voxel_mask = np.asarray(to_numpy(mask)).squeeze().astype(bool)
    voxel_indices = np.argwhere(voxel_mask)
    result = np.zeros((len(points_array), grid.shape[0]), dtype=np.float32)
    if len(voxel_indices) == 0 or len(points_array) == 0:
        return result
    voxel_points = voxel_indices_to_coords(
        voxel_indices.astype(np.float32), grid.shape[1:], coordinate_mode=coordinate_mode, spacing=spacing
    )
    _, nearest = cKDTree(points_array).query(voxel_points)
    np.add.at(result, nearest, grid[:, voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]].T)
    return result


def voxels_to_attributed_points(
    channels: Any,
    mask: Any,
    *,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
    channel_names: Optional[Sequence[str]] = None,
) -> AttributedPoints:
    """One point per voxel inside ``mask``, at the voxel centre, with that voxel's channels (lossless)."""

    grid = as_channel_grid(channels)
    voxel_mask = np.asarray(to_numpy(mask)).squeeze().astype(bool)
    if voxel_mask.shape != grid.shape[1:]:
        raise ValueError(f"mask shape {voxel_mask.shape} does not match grid shape {grid.shape[1:]}.")
    indices = np.argwhere(voxel_mask)
    points = voxel_indices_to_coords(
        indices.astype(np.float32), grid.shape[1:], coordinate_mode=coordinate_mode, spacing=spacing
    )
    features = grid[:, indices[:, 0], indices[:, 1], indices[:, 2]].T
    return make_attributed_points(points, features, channel_names=channel_names)


def subsample_attributed_points(
    samples: AttributedPoints,
    num_points: int,
    *,
    keep: Optional[Any] = None,
    seed: Optional[int] = None,
) -> AttributedPoints:
    """Random subset of ``num_points`` points that always includes the points flagged in ``keep``."""

    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    count = len(samples.points)
    keep_mask = np.zeros(count, dtype=bool) if keep is None else np.asarray(keep, dtype=bool).reshape(-1)
    if keep_mask.shape != (count,):
        raise ValueError(f"keep must have shape ({count},); got {keep_mask.shape}.")
    kept = np.flatnonzero(keep_mask)
    if len(kept) > num_points:
        raise ValueError(f"{len(kept)} points must be kept but only {num_points} were requested.")
    rest = np.flatnonzero(~keep_mask)
    rng = np.random.default_rng(seed)
    fill = rng.choice(rest, size=min(num_points - len(kept), len(rest)), replace=False)
    chosen = np.sort(np.concatenate([kept, fill]))
    return make_attributed_points(
        samples.points[chosen], samples.features[chosen], channel_names=samples.channel_names, metadata=samples.metadata
    )


def gaussian_smooth_channels(
    channels: Any,
    sigma: float | Sequence[float],
    *,
    mode: str = "constant",
) -> np.ndarray:
    """Gaussian-smooth each channel of a grid, turning point quantities into continuous fields."""

    grid = as_channel_grid(channels)
    return np.stack([gaussian_filter(channel, sigma=sigma, mode=mode) for channel in grid]).astype(np.float32, copy=False)


def save_attributed_points(
    samples: AttributedPoints,
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Attributed points can only be saved to .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = make_attributed_points(samples.points, samples.features, channel_names=samples.channel_names)
    np.savez_compressed(
        path, points=samples.points, features=samples.features, channel_names=np.asarray(samples.channel_names)
    )
    sidecar_metadata = metadata or samples.metadata
    if sidecar_metadata is not None and write_sidecar:
        write_metadata(sidecar_metadata, metadata_sidecar_path(path))
    return path


def load_attributed_points(path: str | Path) -> AttributedPoints:
    with np.load(Path(path), allow_pickle=False) as data:
        names = [str(name) for name in data["channel_names"]] if "channel_names" in data else None
        return make_attributed_points(data["points"], data["features"], channel_names=names)


def attributed_points_summary(samples: AttributedPoints) -> dict[str, Any]:
    return {
        "points_shape": tuple(int(value) for value in samples.points.shape),
        "channels": list(samples.channel_names),
        "feature_min": samples.features.min(axis=0).tolist() if len(samples.points) else [],
        "feature_max": samples.features.max(axis=0).tolist() if len(samples.points) else [],
    }
