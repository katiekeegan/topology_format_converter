from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import trimesh

from .mesh import load_mesh
from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata


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


def export_pointcloud(
    points: Any,
    normals: Any,
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    """Export point cloud samples to .npz, .csv, or ASCII .ply."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    points_array = np.asarray(points, dtype=np.float32)
    normals_array = np.asarray(normals, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3]; got {points_array.shape}.")
    if normals_array.shape != points_array.shape:
        raise ValueError(f"normals must have shape {points_array.shape}; got {normals_array.shape}.")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        np.savez_compressed(path, points=points_array, normals=normals_array)
    elif suffix == ".csv":
        values = np.hstack([points_array, normals_array])
        np.savetxt(
            path,
            values,
            delimiter=",",
            header="x,y,z,nx,ny,nz",
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
            handle.write("property float nx\n")
            handle.write("property float ny\n")
            handle.write("property float nz\n")
            handle.write("end_header\n")
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
