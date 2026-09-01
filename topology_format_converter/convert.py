from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np

from .cache import cache_to_mesh, density_to_training_cache, load_training_cache, save_training_cache
from .mesh import convert_mesh, density_to_mesh, export_mesh, load_mesh
from .metadata import ConversionMetadata
from .pointcloud import (
    export_pointcloud,
    load_pointcloud,
    mesh_to_pointcloud,
    pointcloud_to_distance_grid,
    pointcloud_to_occupancy,
)
from .sdf import SDFSamples, load_sdf_samples, save_sdf_samples
from .volume import load_volume, save_signed_distance, save_volume, save_volume_vtk


Modality = Literal[
    "volume",
    "occupancy",
    "sdf-grid",
    "sdf-samples",
    "mesh",
    "pointcloud",
    "training-cache",
]

MESH_EXTENSIONS = {".obj", ".stl", ".ply", ".off", ".glb", ".gltf"}
ARRAY_EXTENSIONS = {".npy", ".npz", ".pt", ".pth"}


def infer_modality(path: str | Path, *, input_key: Optional[str] = None) -> Modality:
    """Infer a source modality from extension and common array keys."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".obj", ".stl", ".off", ".glb", ".gltf"}:
        return "mesh"
    if suffix == ".ply":
        return "mesh"
    if suffix == ".csv":
        return "pointcloud"
    if suffix in {".pt", ".pth", ".npy"}:
        return "volume"
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            keys = set(data.files)
            if {"density", "sdf_grid", "surface_points", "query_points", "query_sdf"}.issubset(keys):
                return "training-cache"
            if {"query_points", "query_sdf"}.issubset(keys) or {"points", "sdf"}.issubset(keys):
                return "sdf-samples"
            if "points" in keys:
                return "pointcloud"
            if (input_key in keys) if input_key is not None else keys.intersection({"density", "theta", "volume", "voxel_grid"}):
                return "volume"
            if "sdf" in keys or "sdf_grid" in keys:
                return "sdf-grid"
            return "volume"
    raise ValueError(f"Cannot infer modality for {path}. Pass source_modality explicitly.")


def modality_from_output(path: str | Path, target_modality: Optional[str] = None) -> Modality:
    if target_modality is not None:
        return target_modality  # type: ignore[return-value]
    suffix = Path(path).suffix.lower()
    if suffix in MESH_EXTENSIONS:
        return "mesh"
    if suffix in {".csv", ".ply"}:
        return "pointcloud"
    if suffix in {".vtk"}:
        return "volume"
    if suffix in {".npz", ".npy"}:
        return "volume"
    raise ValueError(f"Cannot infer target modality for {path}. Pass target_modality explicitly.")


def _shape_from_resolution(resolution: int | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(resolution, int):
        values = (resolution, resolution, resolution)
    else:
        values = tuple(int(value) for value in resolution)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"resolution must be a positive int or three positive ints; got {values}.")
    return values


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    source_modality: Optional[Modality] = None,
    target_modality: Optional[Modality] = None,
    input_key: Optional[str] = None,
    points_key: str = "points",
    normals_key: Optional[str] = "normals",
    sdf_key: str = "query_sdf",
    threshold: float = 0.5,
    coordinate_mode: str = "unit-box",
    spacing: Optional[Iterable[float]] = None,
    resolution: int | Iterable[int] = 64,
    num_points: int = 10000,
    mark_radius: int = 0,
    seed: Optional[int] = None,
    no_repair: bool = False,
    no_metadata: bool = False,
) -> Path:
    """Convert one file between supported topology-shape modalities.

    Explicit `source_modality` and `target_modality` are recommended when an
    extension is ambiguous, for example `.npz` may store a volume, point cloud,
    SDF samples, or a training cache.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    source = source_modality or infer_modality(input_path, input_key=input_key)
    target = modality_from_output(output_path, target_modality)
    metadata = ConversionMetadata(
        source_format=source,
        target_format=target,
        source_path=str(input_path),
        threshold=float(threshold),
        coordinate_mode=coordinate_mode,
    )
    write_sidecar = not no_metadata
    shape = _shape_from_resolution(resolution)

    if source in {"volume", "occupancy"}:
        volume = load_volume(input_path, key=input_key)
        if target in {"volume", "occupancy"}:
            if output_path.suffix.lower() == ".vtk":
                save_volume_vtk(output_path, volume, scalar_name=input_key or "density", spacing=spacing)
                return output_path
            return save_volume(output_path, volume, key="density", metadata=metadata, write_sidecar=write_sidecar)
        if target == "sdf-grid":
            return save_signed_distance(
                output_path,
                volume,
                threshold=threshold,
                spacing=spacing,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "mesh":
            mesh = density_to_mesh(
                volume,
                threshold=threshold,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                repair=not no_repair,
            )
            return export_mesh(mesh, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target == "training-cache":
            cache = density_to_training_cache(
                volume,
                threshold=threshold,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                seed=seed,
                source_format=source,
                source_path=str(input_path),
            )
            return save_training_cache(cache, output_path, write_sidecar=write_sidecar)
        if target == "pointcloud":
            mesh = density_to_mesh(
                volume,
                threshold=threshold,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                repair=not no_repair,
            )
            points, normals = mesh_to_pointcloud(mesh, num_points=num_points, seed=seed)
            return export_pointcloud(points, normals, output_path, metadata=metadata, write_sidecar=write_sidecar)

    if source == "sdf-grid":
        sdf_grid = load_volume(input_path, key=input_key or "sdf")
        if target == "sdf-grid":
            return save_volume(output_path, sdf_grid, key="sdf", metadata=metadata, write_sidecar=write_sidecar)
        if target == "mesh":
            mesh = density_to_mesh(
                sdf_grid,
                threshold=0.0,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                repair=not no_repair,
            )
            return export_mesh(mesh, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target == "pointcloud":
            mesh = density_to_mesh(
                sdf_grid,
                threshold=0.0,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                repair=not no_repair,
            )
            points, normals = mesh_to_pointcloud(mesh, num_points=num_points, seed=seed)
            return export_pointcloud(points, normals, output_path, metadata=metadata, write_sidecar=write_sidecar)

    if source == "mesh":
        mesh = load_mesh(input_path)
        if target == "mesh":
            return convert_mesh(input_path, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target == "pointcloud":
            points, normals = mesh_to_pointcloud(mesh, num_points=num_points, seed=seed)
            return export_pointcloud(points, normals, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target in {"occupancy", "volume", "sdf-grid"}:
            points, _ = mesh_to_pointcloud(mesh, num_points=num_points, seed=seed)
            if target == "sdf-grid":
                distance = pointcloud_to_distance_grid(
                    points,
                    shape=shape,
                    coordinate_mode=coordinate_mode,
                    spacing=spacing,
                )
                return save_volume(output_path, distance, key="sdf", metadata=metadata, write_sidecar=write_sidecar)
            occupancy = pointcloud_to_occupancy(
                points,
                shape=shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                mark_radius=mark_radius,
            )
            return save_volume(output_path, occupancy, key="occupancy", metadata=metadata, write_sidecar=write_sidecar)

    if source == "pointcloud":
        points, normals = load_pointcloud(input_path, points_key=points_key, normals_key=normals_key)
        if target == "pointcloud":
            return export_pointcloud(points, normals, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target in {"occupancy", "volume"}:
            occupancy = pointcloud_to_occupancy(
                points,
                shape=shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                mark_radius=mark_radius,
            )
            return save_volume(output_path, occupancy, key="occupancy", metadata=metadata, write_sidecar=write_sidecar)
        if target == "sdf-grid":
            distance = pointcloud_to_distance_grid(
                points,
                shape=shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
            )
            return save_volume(output_path, distance, key="sdf", metadata=metadata, write_sidecar=write_sidecar)

    if source == "sdf-samples":
        samples = load_sdf_samples(input_path, points_key=points_key, sdf_key=sdf_key, normals_key=normals_key)
        if target == "sdf-samples":
            return save_sdf_samples(samples, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target == "pointcloud":
            return export_pointcloud(samples.points, samples.normals, output_path, metadata=metadata, write_sidecar=write_sidecar)

    if source == "training-cache":
        cache = load_training_cache(input_path)
        if target == "training-cache":
            return save_training_cache(cache, output_path, write_sidecar=write_sidecar)
        if target == "mesh":
            mesh = cache_to_mesh(cache, threshold=threshold, coordinate_mode=coordinate_mode, spacing=spacing, repair=not no_repair)
            return export_mesh(mesh, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target == "pointcloud":
            return export_pointcloud(cache.surface_points, cache.surface_normals, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target == "sdf-samples":
            samples = SDFSamples(cache.query_points, cache.query_sdf, metadata=metadata)
            return save_sdf_samples(samples, output_path, metadata=metadata, write_sidecar=write_sidecar)
        if target in {"volume", "occupancy"}:
            return save_volume(output_path, cache.density, key="density", metadata=metadata, write_sidecar=write_sidecar)
        if target == "sdf-grid":
            return save_volume(output_path, cache.sdf_grid, key="sdf", metadata=metadata, write_sidecar=write_sidecar)

    raise ValueError(f"Unsupported conversion: {source} -> {target}.")
