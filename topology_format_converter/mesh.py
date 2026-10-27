from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from skimage import measure
import trimesh

from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata
from .volume import as_3d_array, parse_spacing, validate_coordinate_mode, voxel_indices_to_coords


def repair_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Apply conservative trimesh cleanup operations."""

    for method_name in (
        "remove_duplicate_faces",
        "remove_degenerate_faces",
        "remove_unreferenced_vertices",
        "remove_infinite_values",
    ):
        method = getattr(mesh, method_name, None)
        if method is not None:
            method()
    try:
        mesh.fill_holes()
    except Exception:
        pass
    try:
        if not mesh.is_winding_consistent:
            mesh.fix_normals()
    except Exception:
        pass
    return mesh


def density_to_mesh(
    density: Any,
    *,
    threshold: float = 0.5,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
    repair: bool = True,
) -> trimesh.Trimesh:
    """Convert a 3-D density volume to a surface mesh with marching cubes."""

    density_array = as_3d_array(density, name="density")
    coordinate_mode = validate_coordinate_mode(coordinate_mode)
    threshold = float(threshold)
    if not (density_array.min() <= threshold <= density_array.max()):
        raise ValueError(
            f"threshold {threshold} is outside density range "
            f"[{density_array.min()}, {density_array.max()}]."
        )
    if density_array.min() == density_array.max():
        raise ValueError("Cannot build a mesh from a constant density volume.")

    mc_spacing = parse_spacing(spacing) or (1.0, 1.0, 1.0)
    verts, faces, _, _ = measure.marching_cubes(density_array, level=threshold, spacing=mc_spacing)
    if coordinate_mode != "voxel":
        verts = voxel_indices_to_coords(verts, density_array.shape, coordinate_mode=coordinate_mode)

    mesh = trimesh.Trimesh(vertices=verts.astype(np.float32), faces=faces.astype(np.int64), process=False)
    mesh = repair_mesh(mesh) if repair else mesh
    # marching_cubes winds faces so that normals point towards decreasing density, i.e. into
    # the solid; flip them so face normals (and sampled point normals) point outward.
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def winding_numbers(mesh: trimesh.Trimesh, points: Any, *, chunk: int = 2048) -> np.ndarray:
    """Generalized winding number of a triangle mesh at each point (about 1 inside, 0 outside).

    Robust to small holes and inconsistent closure, unlike ray-casting containment.
    """

    triangles = np.asarray(mesh.vertices, dtype=np.float64)[np.asarray(mesh.faces)]
    queries = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    values = np.empty(len(queries))
    for start in range(0, len(queries), chunk):
        rel = triangles[None] - queries[start : start + chunk, None, None, :]
        a, b, c = rel[:, :, 0], rel[:, :, 1], rel[:, :, 2]
        la, lb, lc = (np.linalg.norm(v, axis=-1) for v in (a, b, c))
        numerator = np.einsum("nfi,nfi->nf", a, np.cross(b, c))
        denominator = la * lb * lc + np.einsum("nfi,nfi->nf", a, b) * lc + np.einsum("nfi,nfi->nf", b, c) * la \
            + np.einsum("nfi,nfi->nf", c, a) * lb
        values[start : start + chunk] = np.arctan2(numerator, denominator).sum(axis=1) / (2.0 * np.pi)
    return values


def mesh_to_occupancy(
    mesh: trimesh.Trimesh,
    *,
    shape: Iterable[int],
    coordinate_mode: str = "training",
    spacing: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Solid occupancy of a closed (or nearly closed) mesh at voxel centres, by winding number >= 0.5."""

    shape_values = tuple(int(value) for value in shape)
    centres = voxel_indices_to_coords(
        np.argwhere(np.ones(shape_values, dtype=bool)).astype(np.float32), shape_values,
        coordinate_mode=coordinate_mode, spacing=spacing,
    )
    return (winding_numbers(mesh, centres) >= 0.5).astype(np.float32).reshape(shape_values)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    loaded = trimesh.load(Path(path), force="mesh")
    if isinstance(loaded, trimesh.Scene):
        geometries = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not geometries:
            raise ValueError(f"No mesh geometry found in {path}.")
        loaded = trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"Expected a Trimesh from {path}; got {type(loaded).__name__}.")
    return loaded


def export_mesh(
    mesh: trimesh.Trimesh,
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)
    if metadata is not None and write_sidecar:
        write_metadata(metadata, metadata_sidecar_path(path))
    return path


def convert_mesh(
    input_path: str | Path,
    output_path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    mesh = load_mesh(input_path)
    return export_mesh(mesh, output_path, metadata=metadata, write_sidecar=write_sidecar)


def mesh_summary(mesh: trimesh.Trimesh) -> dict[str, Any]:
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "is_watertight": bool(mesh.is_watertight),
        "bounds": np.asarray(mesh.bounds).tolist(),
    }
