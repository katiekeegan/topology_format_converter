from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import trimesh

from .mesh import density_to_mesh, mesh_summary
from .metadata import ConversionMetadata, write_metadata
from .volume import (
    as_3d_array,
    coordinate_bounds,
    parse_spacing,
    sample_sdf_grid,
    sdf_normalization_scale,
    signed_distance_from_density,
    validate_coordinate_mode,
)


@dataclass
class TrainingCache:
    """Derived arrays useful for SDF/VAE training."""

    density: np.ndarray
    sdf_grid: np.ndarray
    surface_points: np.ndarray
    surface_normals: np.ndarray
    query_points: np.ndarray
    query_sdf: np.ndarray
    metadata: ConversionMetadata


def _sample_surface(mesh: trimesh.Trimesh, num_surface_points: int, seed: Optional[int]) -> tuple[np.ndarray, np.ndarray]:
    if num_surface_points <= 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)

    state = None
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, num_surface_points)
    finally:
        if state is not None:
            np.random.set_state(state)

    normals = mesh.face_normals[face_indices]
    return points.astype(np.float32, copy=False), normals.astype(np.float32, copy=False)


def _sample_query_points(
    surface_points: np.ndarray,
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    num_query_points: int,
    surface_sample_ratio: float,
    noise_std: float,
    seed: Optional[int],
) -> np.ndarray:
    if not 0.0 <= surface_sample_ratio <= 1.0:
        raise ValueError("surface_sample_ratio must be between 0 and 1.")
    if num_query_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)
    n_surface = int(num_query_points * surface_sample_ratio)
    n_uniform = num_query_points - n_surface

    if n_surface > 0 and len(surface_points) > 0:
        idx = rng.integers(0, len(surface_points), size=n_surface)
        near_surface = surface_points[idx] + rng.normal(0.0, noise_std, size=(n_surface, 3))
    else:
        near_surface = np.empty((0, 3), dtype=np.float32)

    uniform = rng.uniform(bounds_min, bounds_max, size=(n_uniform, 3)) if n_uniform > 0 else np.empty((0, 3))
    points = np.vstack([near_surface, uniform]).astype(np.float32, copy=False)
    if len(points) > 0:
        rng.shuffle(points, axis=0)
    return np.clip(points, bounds_min, bounds_max).astype(np.float32, copy=False)


def density_to_training_cache(
    density: Any,
    *,
    threshold: float = 0.5,
    coordinate_mode: str = "training",
    spacing: Optional[Iterable[float]] = None,
    num_surface_points: int = 10000,
    num_query_points: int = 10000,
    surface_sample_ratio: float = 0.7,
    noise_std: float = 0.05,
    normalize_sdf: bool = True,
    seed: Optional[int] = None,
    source_format: str = "volume",
    source_path: Optional[str] = None,
    metadata_extra: Optional[dict[str, Any]] = None,
) -> TrainingCache:
    """Build a reusable training cache from a density volume."""

    density_array = as_3d_array(density, name="density")
    coordinate_mode = validate_coordinate_mode(coordinate_mode)
    spacing_values = parse_spacing(spacing)
    mesh = density_to_mesh(
        density_array,
        threshold=threshold,
        coordinate_mode=coordinate_mode,
        spacing=spacing_values,
        repair=True,
    )
    sdf_grid = signed_distance_from_density(density_array, threshold=threshold, spacing=spacing_values)
    scale = sdf_normalization_scale(density_array.shape, spacing=spacing_values)
    if normalize_sdf and scale > 0:
        sdf_grid = sdf_grid / scale

    surface_points, surface_normals = _sample_surface(mesh, num_surface_points, seed)
    bounds_min, bounds_max = coordinate_bounds(
        density_array.shape,
        coordinate_mode=coordinate_mode,
        spacing=spacing_values,
    )
    query_points = _sample_query_points(
        surface_points,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        num_query_points=num_query_points,
        surface_sample_ratio=surface_sample_ratio,
        noise_std=noise_std,
        seed=seed,
    )
    query_sdf = sample_sdf_grid(
        sdf_grid,
        query_points,
        coordinate_mode=coordinate_mode,
        spacing=spacing_values,
    )

    metadata = ConversionMetadata(
        source_format=source_format,
        target_format="training-cache",
        source_path=source_path,
        threshold=float(threshold),
        voxel_shape=tuple(int(value) for value in density_array.shape),
        spacing=spacing_values,
        coordinate_mode=coordinate_mode,
        extra={
            "num_surface_points": int(num_surface_points),
            "num_query_points": int(num_query_points),
            "surface_sample_ratio": float(surface_sample_ratio),
            "noise_std": float(noise_std),
            "normalize_sdf": bool(normalize_sdf),
            **(metadata_extra or {}),
        },
    )

    return TrainingCache(
        density=density_array.astype(np.float32, copy=False),
        sdf_grid=sdf_grid.astype(np.float32, copy=False),
        surface_points=surface_points,
        surface_normals=surface_normals,
        query_points=query_points,
        query_sdf=query_sdf.astype(np.float32, copy=False),
        metadata=metadata,
    )


def save_training_cache(cache: TrainingCache, path: str | Path, *, write_sidecar: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(cache.metadata.to_dict(), sort_keys=True)
    np.savez_compressed(
        path,
        density=cache.density,
        sdf_grid=cache.sdf_grid,
        surface_points=cache.surface_points,
        surface_normals=cache.surface_normals,
        query_points=cache.query_points,
        query_sdf=cache.query_sdf,
        metadata_json=np.asarray(metadata_json),
    )
    if write_sidecar:
        write_metadata(cache.metadata, path.with_suffix(path.suffix + ".json"))
    return path


def load_training_cache(path: str | Path) -> TrainingCache:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        metadata_raw = str(data["metadata_json"].item())
        metadata = ConversionMetadata.from_dict(json.loads(metadata_raw))
        return TrainingCache(
            density=data["density"].astype(np.float32, copy=False),
            sdf_grid=data["sdf_grid"].astype(np.float32, copy=False),
            surface_points=data["surface_points"].astype(np.float32, copy=False),
            surface_normals=data["surface_normals"].astype(np.float32, copy=False),
            query_points=data["query_points"].astype(np.float32, copy=False),
            query_sdf=data["query_sdf"].astype(np.float32, copy=False),
            metadata=metadata,
        )


def cache_to_mesh(
    cache: TrainingCache,
    *,
    threshold: Optional[float] = None,
    coordinate_mode: Optional[str] = None,
    spacing: Optional[Iterable[float]] = None,
    repair: bool = True,
):
    threshold_value = threshold if threshold is not None else cache.metadata.threshold
    if threshold_value is None:
        threshold_value = 0.5
    coordinate_mode_value = coordinate_mode or cache.metadata.coordinate_mode or "training"
    spacing_value = spacing if spacing is not None else cache.metadata.spacing
    return density_to_mesh(
        cache.density,
        threshold=threshold_value,
        coordinate_mode=coordinate_mode_value,
        spacing=spacing_value,
        repair=repair,
    )


def cache_summary(cache: TrainingCache) -> dict[str, Any]:
    mesh = None
    try:
        mesh = mesh_summary(cache_to_mesh(cache))
    except Exception as exc:
        mesh = {"error": str(exc)}
    return {
        "density_shape": tuple(int(value) for value in cache.density.shape),
        "sdf_grid_shape": tuple(int(value) for value in cache.sdf_grid.shape),
        "surface_points_shape": tuple(int(value) for value in cache.surface_points.shape),
        "surface_normals_shape": tuple(int(value) for value in cache.surface_normals.shape),
        "query_points_shape": tuple(int(value) for value in cache.query_points.shape),
        "query_sdf_shape": tuple(int(value) for value in cache.query_sdf.shape),
        "density_min": float(np.min(cache.density)),
        "density_max": float(np.max(cache.density)),
        "sdf_min": float(np.min(cache.sdf_grid)),
        "sdf_max": float(np.max(cache.sdf_grid)),
        "metadata": cache.metadata.to_dict(),
        "mesh": mesh,
    }
