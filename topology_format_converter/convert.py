from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal, Optional
import warnings

import numpy as np

from .cache import (
    cache_to_mesh,
    density_to_training_cache,
    load_training_cache,
    save_training_cache,
)
from .field import FieldSamples, load_field_samples, save_field_samples
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
from .sparse import (
    dense_to_sparse,
    load_sparse_voxels,
    save_sparse_voxels,
    sparse_to_dense,
)
from .volume import (
    load_volume,
    save_volume,
    save_volume_vtk,
    signed_distance_from_density,
    truncate_distance_grid,
    voxel_indices_to_coords,
)

Modality = Literal[
    "volume",
    "occupancy",
    "sdf-grid",
    "udf-grid",
    "tsdf-grid",
    "sdf-samples",
    "field-samples",
    "sparse-voxels",
    "mesh",
    "pointcloud",
    "training-cache",
]

MESH_EXTENSIONS = {".obj", ".stl", ".ply", ".off", ".glb", ".gltf"}
ARRAY_EXTENSIONS = {".npy", ".npz", ".pt", ".pth"}
UNSIGNED_DISTANCE_WARNING = (
    "Converting a mesh or surface point cloud to sdf-grid writes an unsigned "
    "nearest-surface distance grid because the source does not encode reliable "
    "inside/outside sign information."
)
UNSIGNED_TO_SDF_WARNING = (
    "Converting an unsigned distance field to sdf-grid cannot recover inside/outside "
    "sign information, so the output remains unsigned."
)


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
            if {
                "density",
                "sdf_grid",
                "surface_points",
                "query_points",
                "query_sdf",
            }.issubset(keys):
                return "training-cache"
            if {"indices", "shape"}.issubset(keys) or {"coords", "grid_shape"}.issubset(
                keys
            ):
                return "sparse-voxels"
            if {"query_points", "query_sdf"}.issubset(keys) or {
                "points",
                "sdf",
            }.issubset(keys):
                return "sdf-samples"
            if "points" in keys and keys.intersection(
                {"values", "value", "signed_distance", "occupancy", "density", "stress"}
            ):
                return "field-samples"
            if "points" in keys:
                return "pointcloud"
            if "udf" in keys or "udf_grid" in keys:
                return "udf-grid"
            if "tsdf" in keys or "tsdf_grid" in keys:
                return "tsdf-grid"
            if "sdf" in keys or "sdf_grid" in keys:
                return "sdf-grid"
            if "occupancy" in keys:
                return "occupancy"
            if (
                (input_key in keys)
                if input_key is not None
                else keys.intersection({"density", "theta", "volume", "voxel_grid"})
            ):
                return "volume"
            return "volume"
    raise ValueError(
        f"Cannot infer modality for {path}. Pass source_modality explicitly."
    )


def modality_from_output(
    path: str | Path, target_modality: Optional[str] = None
) -> Modality:
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
    raise ValueError(
        f"Cannot infer target modality for {path}. Pass target_modality explicitly."
    )


def _shape_from_resolution(resolution: int | Iterable[int]) -> tuple[int, int, int]:
    if isinstance(resolution, int):
        values = (resolution, resolution, resolution)
    else:
        values = tuple(int(value) for value in resolution)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(
            f"resolution must be a positive int or three positive ints; got {values}."
        )
    return values


def _mark_unsigned_distance(metadata: ConversionMetadata) -> None:
    warnings.warn(UNSIGNED_DISTANCE_WARNING, UserWarning, stacklevel=3)
    metadata.sdf_sign = "unsigned"
    metadata.extra["distance_kind"] = "unsigned_nearest_surface"
    metadata.extra["warning"] = UNSIGNED_DISTANCE_WARNING


def _mark_unsigned_to_sdf(metadata: ConversionMetadata) -> None:
    warnings.warn(UNSIGNED_TO_SDF_WARNING, UserWarning, stacklevel=3)
    metadata.sdf_sign = "unsigned"
    metadata.extra["distance_kind"] = "unsigned_distance"
    metadata.extra["warning"] = UNSIGNED_TO_SDF_WARNING


def _distance_key(modality: str) -> str:
    if modality == "sdf-grid":
        return "sdf"
    if modality == "udf-grid":
        return "udf"
    if modality == "tsdf-grid":
        return "tsdf"
    raise ValueError(f"{modality!r} is not a distance-grid modality.")


def _save_distance_grid(
    output_path: Path,
    grid: np.ndarray,
    target: str,
    metadata: ConversionMetadata,
    *,
    truncation: float,
    write_sidecar: bool,
) -> Path:
    if target == "sdf-grid":
        return save_volume(
            output_path, grid, key="sdf", metadata=metadata, write_sidecar=write_sidecar
        )
    if target == "udf-grid":
        metadata.sdf_sign = "unsigned"
        metadata.extra["distance_kind"] = "unsigned_distance"
        return save_volume(
            output_path,
            np.abs(grid),
            key="udf",
            metadata=metadata,
            write_sidecar=write_sidecar,
        )
    if target == "tsdf-grid":
        metadata.extra["distance_kind"] = "truncated_signed_distance"
        metadata.extra["truncation"] = float(truncation)
        return save_volume(
            output_path,
            truncate_distance_grid(grid, truncation),
            key="tsdf",
            metadata=metadata,
            write_sidecar=write_sidecar,
        )
    raise ValueError(f"{target!r} is not a distance-grid target.")


def _grid_to_field_samples(
    grid: np.ndarray,
    *,
    coordinate_mode: str,
    spacing: Optional[Iterable[float]],
    field_name: str,
    metadata: ConversionMetadata,
) -> FieldSamples:
    indices = np.indices(grid.shape, dtype=np.float32).reshape(3, -1).T
    points = voxel_indices_to_coords(
        indices, grid.shape, coordinate_mode=coordinate_mode, spacing=spacing
    )
    return FieldSamples(
        points=points, values=grid.reshape(-1), field_name=field_name, metadata=metadata
    )


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
    truncation: float = 0.1,
    values_key: str = "values",
    field_name: Optional[str] = None,
    seed: Optional[int] = None,
    no_repair: bool = False,
    no_metadata: bool = False,
) -> Path:
    """Convert one file between supported topology-shape modalities.

    Explicit `source_modality` and `target_modality` are recommended when an
    extension is ambiguous, for example `.npz` may store a volume, point cloud,
    SDF samples, field samples, sparse voxels, or a training cache.
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
                save_volume_vtk(
                    output_path,
                    volume,
                    scalar_name=input_key or "density",
                    spacing=spacing,
                )
                return output_path
            return save_volume(
                output_path,
                volume,
                key="occupancy" if target == "occupancy" else "density",
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target in {"sdf-grid", "udf-grid", "tsdf-grid"}:
            sdf_grid = signed_distance_from_density(
                volume, threshold=threshold, spacing=spacing
            )
            return _save_distance_grid(
                output_path,
                sdf_grid,
                target,
                metadata,
                truncation=truncation,
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
            return export_mesh(
                mesh, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
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
            return export_pointcloud(
                points,
                normals,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "sparse-voxels":
            sparse = dense_to_sparse(
                volume, threshold=threshold if source == "occupancy" else None
            )
            return save_sparse_voxels(
                sparse, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "field-samples":
            samples = _grid_to_field_samples(
                volume,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                field_name=input_key or source,
                metadata=metadata,
            )
            return save_field_samples(
                samples, output_path, metadata=metadata, write_sidecar=write_sidecar
            )

    if source in {"sdf-grid", "udf-grid", "tsdf-grid"}:
        source_key = input_key or _distance_key(source)
        distance_grid = load_volume(input_path, key=input_key)
        if target in {"sdf-grid", "udf-grid", "tsdf-grid"}:
            if source == "udf-grid" and target in {"sdf-grid", "tsdf-grid"}:
                if target == "sdf-grid":
                    _mark_unsigned_to_sdf(metadata)
                    return save_volume(
                        output_path,
                        distance_grid,
                        key="sdf",
                        metadata=metadata,
                        write_sidecar=write_sidecar,
                    )
                raise ValueError(
                    "Unsupported conversion: udf-grid -> tsdf-grid because sign information is unavailable."
                )
            return _save_distance_grid(
                output_path,
                distance_grid,
                target,
                metadata,
                truncation=truncation,
                write_sidecar=write_sidecar,
            )
        if target == "mesh":
            if source == "udf-grid":
                raise ValueError(
                    "Unsupported conversion: udf-grid -> mesh because unsigned distances do not define inside/outside."
                )
            mesh = density_to_mesh(
                distance_grid,
                threshold=0.0,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                repair=not no_repair,
            )
            return export_mesh(
                mesh, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "pointcloud":
            if source == "udf-grid":
                raise ValueError(
                    "Unsupported conversion: udf-grid -> pointcloud because unsigned distances do not define a signed zero level."
                )
            mesh = density_to_mesh(
                distance_grid,
                threshold=0.0,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                repair=not no_repair,
            )
            points, normals = mesh_to_pointcloud(mesh, num_points=num_points, seed=seed)
            return export_pointcloud(
                points,
                normals,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "sparse-voxels":
            sparse = dense_to_sparse(distance_grid)
            return save_sparse_voxels(
                sparse, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "field-samples":
            samples = _grid_to_field_samples(
                distance_grid,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                field_name=source_key,
                metadata=metadata,
            )
            return save_field_samples(
                samples, output_path, metadata=metadata, write_sidecar=write_sidecar
            )

    if source == "mesh":
        mesh = load_mesh(input_path)
        if target == "mesh":
            return convert_mesh(
                input_path, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "pointcloud":
            points, normals = mesh_to_pointcloud(mesh, num_points=num_points, seed=seed)
            return export_pointcloud(
                points,
                normals,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target in {"occupancy", "volume", "sdf-grid", "udf-grid", "sparse-voxels"}:
            points, _ = mesh_to_pointcloud(mesh, num_points=num_points, seed=seed)
            if target == "sdf-grid":
                _mark_unsigned_distance(metadata)
                distance = pointcloud_to_distance_grid(
                    points,
                    shape=shape,
                    coordinate_mode=coordinate_mode,
                    spacing=spacing,
                )
                return save_volume(
                    output_path,
                    distance,
                    key="sdf",
                    metadata=metadata,
                    write_sidecar=write_sidecar,
                )
            if target == "udf-grid":
                distance = pointcloud_to_distance_grid(
                    points,
                    shape=shape,
                    coordinate_mode=coordinate_mode,
                    spacing=spacing,
                )
                metadata.sdf_sign = "unsigned"
                metadata.extra["distance_kind"] = "unsigned_nearest_surface"
                return save_volume(
                    output_path,
                    distance,
                    key="udf",
                    metadata=metadata,
                    write_sidecar=write_sidecar,
                )
            occupancy = pointcloud_to_occupancy(
                points,
                shape=shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                mark_radius=mark_radius,
            )
            if target == "sparse-voxels":
                sparse = dense_to_sparse(occupancy)
                return save_sparse_voxels(
                    sparse, output_path, metadata=metadata, write_sidecar=write_sidecar
                )
            return save_volume(
                output_path,
                occupancy,
                key="occupancy",
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "tsdf-grid":
            raise ValueError(
                "Unsupported conversion: mesh -> tsdf-grid because sign information is unavailable. Use udf-grid for unsigned distance."
            )

    if source == "pointcloud":
        points, normals = load_pointcloud(
            input_path, points_key=points_key, normals_key=normals_key
        )
        if target == "pointcloud":
            return export_pointcloud(
                points,
                normals,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target in {"occupancy", "volume"}:
            occupancy = pointcloud_to_occupancy(
                points,
                shape=shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                mark_radius=mark_radius,
            )
            return save_volume(
                output_path,
                occupancy,
                key="occupancy",
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "sdf-grid":
            _mark_unsigned_distance(metadata)
            distance = pointcloud_to_distance_grid(
                points,
                shape=shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
            )
            return save_volume(
                output_path,
                distance,
                key="sdf",
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "udf-grid":
            distance = pointcloud_to_distance_grid(
                points,
                shape=shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
            )
            metadata.sdf_sign = "unsigned"
            metadata.extra["distance_kind"] = "unsigned_nearest_surface"
            return save_volume(
                output_path,
                distance,
                key="udf",
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "sparse-voxels":
            occupancy = pointcloud_to_occupancy(
                points,
                shape=shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                mark_radius=mark_radius,
            )
            sparse = dense_to_sparse(occupancy)
            return save_sparse_voxels(
                sparse, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "field-samples":
            samples = FieldSamples(
                points=points,
                values=np.zeros((len(points),), dtype=np.float32),
                field_name=field_name or "value",
                normals=normals,
                metadata=metadata,
            )
            metadata.extra["field_values"] = "zeros"
            return save_field_samples(
                samples, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "tsdf-grid":
            raise ValueError(
                "Unsupported conversion: pointcloud -> tsdf-grid because sign information is unavailable. Use udf-grid for unsigned distance."
            )

    if source == "sdf-samples":
        samples = load_sdf_samples(
            input_path, points_key=points_key, sdf_key=sdf_key, normals_key=normals_key
        )
        if target == "sdf-samples":
            return save_sdf_samples(
                samples, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "pointcloud":
            return export_pointcloud(
                samples.points,
                samples.normals,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "field-samples":
            field_samples = FieldSamples(
                samples.points,
                samples.sdf,
                field_name="sdf",
                normals=samples.normals,
                metadata=metadata,
            )
            return save_field_samples(
                field_samples,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )

    if source == "field-samples":
        samples = load_field_samples(
            input_path,
            points_key=points_key,
            values_key=values_key,
            normals_key=normals_key,
            field_name=field_name,
        )
        if target == "field-samples":
            return save_field_samples(
                samples, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "pointcloud":
            return export_pointcloud(
                samples.points,
                samples.normals,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "sdf-samples":
            sdf_samples = SDFSamples(
                samples.points,
                samples.values,
                normals=samples.normals,
                metadata=metadata,
            )
            return save_sdf_samples(
                sdf_samples, output_path, metadata=metadata, write_sidecar=write_sidecar
            )

    if source == "sparse-voxels":
        sparse = load_sparse_voxels(input_path)
        dense = sparse_to_dense(sparse)
        if target == "sparse-voxels":
            return save_sparse_voxels(
                sparse, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target in {"volume", "occupancy"}:
            return save_volume(
                output_path,
                dense,
                key="occupancy" if target == "occupancy" else "density",
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target in {"sdf-grid", "udf-grid", "tsdf-grid"}:
            sdf_grid = signed_distance_from_density(
                dense, threshold=threshold, spacing=spacing
            )
            return _save_distance_grid(
                output_path,
                sdf_grid,
                target,
                metadata,
                truncation=truncation,
                write_sidecar=write_sidecar,
            )
        if target == "mesh":
            mesh = density_to_mesh(
                dense,
                threshold=threshold,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                repair=not no_repair,
            )
            return export_mesh(
                mesh, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "pointcloud":
            points = voxel_indices_to_coords(
                sparse.indices,
                sparse.shape,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
            )
            return export_pointcloud(
                points,
                None,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "training-cache":
            cache = density_to_training_cache(
                dense,
                threshold=threshold,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                seed=seed,
                source_format=source,
                source_path=str(input_path),
            )
            return save_training_cache(cache, output_path, write_sidecar=write_sidecar)

    if source == "training-cache":
        cache = load_training_cache(input_path)
        if target == "training-cache":
            return save_training_cache(cache, output_path, write_sidecar=write_sidecar)
        if target == "mesh":
            mesh = cache_to_mesh(
                cache,
                threshold=threshold,
                coordinate_mode=coordinate_mode,
                spacing=spacing,
                repair=not no_repair,
            )
            return export_mesh(
                mesh, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "pointcloud":
            return export_pointcloud(
                cache.surface_points,
                cache.surface_normals,
                output_path,
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target == "sdf-samples":
            samples = SDFSamples(cache.query_points, cache.query_sdf, metadata=metadata)
            return save_sdf_samples(
                samples, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target == "field-samples":
            samples = FieldSamples(
                cache.query_points, cache.query_sdf, field_name="sdf", metadata=metadata
            )
            return save_field_samples(
                samples, output_path, metadata=metadata, write_sidecar=write_sidecar
            )
        if target in {"volume", "occupancy"}:
            return save_volume(
                output_path,
                cache.density,
                key="occupancy" if target == "occupancy" else "density",
                metadata=metadata,
                write_sidecar=write_sidecar,
            )
        if target in {"sdf-grid", "udf-grid", "tsdf-grid"}:
            return _save_distance_grid(
                output_path,
                cache.sdf_grid,
                target,
                metadata,
                truncation=truncation,
                write_sidecar=write_sidecar,
            )
        if target == "sparse-voxels":
            sparse = dense_to_sparse(cache.density, threshold=threshold)
            return save_sparse_voxels(
                sparse, output_path, metadata=metadata, write_sidecar=write_sidecar
            )

    raise ValueError(f"Unsupported conversion: {source} -> {target}.")
