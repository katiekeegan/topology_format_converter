from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from scipy.spatial import cKDTree
import trimesh

from .mesh import load_mesh
from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata
from .volume import coords_to_voxel_indices, voxel_indices_to_coords


def mesh_to_pointcloud(
    mesh: trimesh.Trimesh,
    *,
    num_points: int = 10000,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample surface points and face normals from a mesh."""

    if num_points <= 0:
        raise ValueError("num_points must be positive.")
    state = None
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
    try:
        points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    finally:
        if state is not None:
            np.random.set_state(state)
    normals = mesh.face_normals[face_indices]
    return points.astype(np.float32, copy=False), normals.astype(np.float32, copy=False)


def load_mesh_as_pointcloud(
    path: str | Path,
    *,
    num_points: int = 10000,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray]:
    return mesh_to_pointcloud(load_mesh(path), num_points=num_points, seed=seed)


def _validate_points(points: Any) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3]; got {points_array.shape}.")
    return points_array


def _validate_normals(normals: Any, points_shape: tuple[int, int]) -> np.ndarray:
    normals_array = np.asarray(normals, dtype=np.float32)
    if normals_array.shape != points_shape:
        raise ValueError(f"normals must have shape {points_shape}; got {normals_array.shape}.")
    return normals_array


def load_pointcloud(
    path: str | Path,
    *,
    points_key: str = "points",
    normals_key: Optional[str] = "normals",
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Load point cloud data from .npz, .csv, or .ply.

    CSV input can either have a header with x,y,z and optional nx,ny,nz columns,
    or plain numeric columns where the first three columns are point positions
    and the next three columns, when present, are normals.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if points_key not in data:
                raise KeyError(f"Could not find points key {points_key!r}. Available keys: {data.files}")
            points = _validate_points(data[points_key])
            normals = None
            if normals_key and normals_key in data:
                normals = _validate_normals(data[normals_key], points.shape)
            return points, normals

    if suffix == ".csv":
        try:
            named = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
            if named.dtype.names and {"x", "y", "z"}.issubset(named.dtype.names):
                points = np.column_stack([named["x"], named["y"], named["z"]]).astype(np.float32, copy=False)
                normals = None
                if {"nx", "ny", "nz"}.issubset(named.dtype.names):
                    normals = np.column_stack([named["nx"], named["ny"], named["nz"]]).astype(np.float32, copy=False)
                return _validate_points(points), normals
        except ValueError:
            pass
        values = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] < 3:
            raise ValueError(f"CSV point cloud must have at least 3 columns; got {values.shape[1]}.")
        points = _validate_points(values[:, :3])
        normals = _validate_normals(values[:, 3:6], points.shape) if values.shape[1] >= 6 else None
        return points, normals

    if suffix == ".ply":
        loaded = trimesh.load(path)
        if isinstance(loaded, trimesh.PointCloud):
            points = _validate_points(loaded.vertices)
            return points, None
        if isinstance(loaded, trimesh.Trimesh):
            points = _validate_points(loaded.vertices)
            normals = np.asarray(loaded.vertex_normals, dtype=np.float32)
            if normals.shape != points.shape:
                normals = None
            return points, normals
        raise TypeError(f"Expected a point cloud or mesh from {path}; got {type(loaded).__name__}.")

    raise ValueError(f"Unsupported point-cloud extension {suffix!r}. Use .npz, .csv, or .ply.")


def export_pointcloud(
    points: Any,
    normals: Optional[Any],
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    """Export point cloud samples to .npz, .csv, or ASCII .ply."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points_array = _validate_points(points)
    normals_array = None if normals is None else _validate_normals(normals, points_array.shape)

    suffix = path.suffix.lower()
    if suffix == ".npz":
        arrays = {"points": points_array}
        if normals_array is not None:
            arrays["normals"] = normals_array
        np.savez_compressed(path, **arrays)
    elif suffix == ".csv":
        if normals_array is None:
            values = points_array
            header = "x,y,z"
        else:
            values = np.hstack([points_array, normals_array])
            header = "x,y,z,nx,ny,nz"
        np.savetxt(
            path,
            values,
            delimiter=",",
            header=header,
            comments="",
        )
    elif suffix == ".ply":
        with path.open("w", encoding="utf-8") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {len(points_array)}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            if normals_array is not None:
                handle.write("property float nx\n")
                handle.write("property float ny\n")
                handle.write("property float nz\n")
            handle.write("end_header\n")
            if normals_array is None:
                for point in points_array:
                    handle.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g}\n")
            else:
                for point, normal in zip(points_array, normals_array):
                    handle.write(
                        f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                        f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g}\n"
                    )
    else:
        raise ValueError(f"Unsupported point-cloud extension {suffix!r}. Use .npz, .csv, or .ply.")

    if metadata is not None and write_sidecar:
        write_metadata(metadata, metadata_sidecar_path(path))
    return path


def pointcloud_to_occupancy(
    points: Any,
    *,
    shape: Iterable[int] = (64, 64, 64),
    coordinate_mode: str = "unit-box",
    spacing: Optional[Iterable[float]] = None,
    mark_radius: int = 0,
) -> np.ndarray:
    """Rasterize point positions into a binary surface-occupancy grid."""

    points_array = _validate_points(points)
    shape_values = tuple(int(value) for value in shape)
    if len(shape_values) != 3 or any(value <= 0 for value in shape_values):
        raise ValueError(f"shape must contain three positive integers; got {shape_values}.")
    if mark_radius < 0:
        raise ValueError("mark_radius must be non-negative.")

    indices = np.rint(
        coords_to_voxel_indices(
            points_array,
            shape_values,
            coordinate_mode=coordinate_mode,
            spacing=spacing,
        )
    ).astype(np.int64)
    occupancy = np.zeros(shape_values, dtype=np.float32)
    valid = np.all((indices >= 0) & (indices < np.asarray(shape_values)), axis=1)
    for index in indices[valid]:
        x, y, z = (int(value) for value in index)
        x0, x1 = max(0, x - mark_radius), min(shape_values[0], x + mark_radius + 1)
        y0, y1 = max(0, y - mark_radius), min(shape_values[1], y + mark_radius + 1)
        z0, z1 = max(0, z - mark_radius), min(shape_values[2], z + mark_radius + 1)
        occupancy[x0:x1, y0:y1, z0:z1] = 1.0
    return occupancy


def pointcloud_to_distance_grid(
    points: Any,
    *,
    shape: Iterable[int] = (64, 64, 64),
    coordinate_mode: str = "unit-box",
    spacing: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Convert a surface point cloud to an unsigned nearest-surface distance grid."""

    points_array = _validate_points(points)
    shape_values = tuple(int(value) for value in shape)
    if len(shape_values) != 3 or any(value <= 0 for value in shape_values):
        raise ValueError(f"shape must contain three positive integers; got {shape_values}.")
    tree = cKDTree(points_array)
    grid_indices = np.indices(shape_values, dtype=np.float32).reshape(3, -1).T
    grid_points = voxel_indices_to_coords(
        grid_indices,
        shape_values,
        coordinate_mode=coordinate_mode,
        spacing=spacing,
    )
    distances, _ = tree.query(grid_points)
    return distances.reshape(shape_values).astype(np.float32, copy=False)


def pointcloud_summary(points: Any, normals: Optional[Any] = None) -> dict[str, Any]:
    points_array = _validate_points(points)
    result: dict[str, Any] = {
        "points": int(len(points_array)),
        "bounds": [points_array.min(axis=0).tolist(), points_array.max(axis=0).tolist()],
    }
    if normals is not None:
        normals_array = _validate_normals(normals, points_array.shape)
        result["has_normals"] = True
        result["normal_mean_norm"] = float(np.linalg.norm(normals_array, axis=1).mean())
    else:
        result["has_normals"] = False
    return result
