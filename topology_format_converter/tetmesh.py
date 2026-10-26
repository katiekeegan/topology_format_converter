from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay, cKDTree
from skimage import measure
import trimesh

from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata
from .volume import as_3d_array, threshold_density, validate_coordinate_mode, voxel_indices_to_coords


@dataclass
class TetMesh:
    """Volumetric tetrahedral mesh: vertex positions and ``[T, 4]`` vertex indices per tetrahedron."""

    vertices: np.ndarray
    tetrahedra: np.ndarray
    metadata: Optional[ConversionMetadata] = None


def make_tet_mesh(vertices: Any, tetrahedra: Any, *, metadata: Optional[ConversionMetadata] = None) -> TetMesh:
    vertices_array = np.asarray(vertices, dtype=np.float32)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError(f"vertices must have shape [V, 3]; got {vertices_array.shape}.")
    tets = np.asarray(tetrahedra, dtype=np.int64).reshape(-1, 4)
    if len(tets) and (tets.min() < 0 or tets.max() >= len(vertices_array)):
        raise ValueError("tetrahedra reference vertices outside the mesh.")
    return TetMesh(vertices=vertices_array, tetrahedra=tets, metadata=metadata)


def tet_mesh_volumes(mesh: TetMesh) -> np.ndarray:
    """Signed volume of every tetrahedron."""

    a, b, c, d = (mesh.vertices[mesh.tetrahedra[:, index]].astype(np.float64) for index in range(4))
    return (np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0).astype(np.float32)


def density_to_tet_mesh(
    density: Any,
    *,
    threshold: float = 0.5,
    interior_spacing: int = 2,
    extra_points: Optional[Any] = None,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
    min_volume: float = 1e-3,
) -> TetMesh:
    """Tetrahedralize the solid region of a density grid (an unstructured volume mesh).

    Vertices are the marching-cubes surface vertices of the region, the voxel
    centres inside it on a lattice of stride ``interior_spacing``, and any
    ``extra_points`` (voxel-index coordinates; e.g. load or support locations
    that should be mesh nodes). The points are Delaunay-tetrahedralized and a
    tetrahedron is kept when the trilinearly interpolated occupancy at its
    centroid is at least 0.5 and its volume is at least ``min_volume`` cubic
    voxels, which removes the convex-hull fill outside non-convex regions and
    degenerate slivers. The mesh fills the region's marching-cubes volume, but
    near-flat tetrahedra on curved boundaries mean its boundary surface is not
    guaranteed to be watertight.
    """

    density_array = as_3d_array(density, name="density")
    coordinate_mode = validate_coordinate_mode(coordinate_mode)
    if int(interior_spacing) < 1:
        raise ValueError("interior_spacing must be a positive integer.")
    occupancy = threshold_density(density_array, threshold=threshold).astype(np.float32)
    if occupancy.sum() < 4:
        raise ValueError("The solid region needs at least four voxels to tetrahedralize.")

    surface, _, _, _ = measure.marching_cubes(np.pad(occupancy, 1), level=0.5)
    interior = np.argwhere(occupancy > 0)
    interior = interior[np.all(interior % int(interior_spacing) == 0, axis=1)]
    candidates = [surface - 1.0, interior.astype(np.float32)]
    if extra_points is not None:
        candidates.append(np.asarray(extra_points, dtype=np.float32).reshape(-1, 3))
    points = np.unique(np.round(np.concatenate(candidates), 4), axis=0)

    tets = Delaunay(points, qhull_options="QJ").simplices.astype(np.int64)
    centroids = points[tets].mean(axis=1)
    # Pad so centroids in the half-voxel band outside the outer voxel centres interpolate
    # towards empty space instead of being treated as outside the grid.
    inside = map_coordinates(np.pad(occupancy, 1), (centroids + 1.0).T, order=1, mode="nearest") >= 0.5
    mesh = make_tet_mesh(points, tets[inside])
    volumes = tet_mesh_volumes(mesh)
    tets = mesh.tetrahedra[np.abs(volumes) >= float(min_volume)]
    # Orient every tetrahedron positively and drop vertices no tetrahedron uses.
    flip = tet_mesh_volumes(make_tet_mesh(points, tets)) < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    used, tets = np.unique(tets, return_inverse=True)
    vertices = voxel_indices_to_coords(points[used], density_array.shape, coordinate_mode=coordinate_mode, spacing=spacing)
    metadata = ConversionMetadata(
        source_format="volume",
        target_format="tet-mesh",
        threshold=float(threshold),
        voxel_shape=tuple(int(value) for value in density_array.shape),
        coordinate_mode=coordinate_mode,
        extra={"interior_spacing": int(interior_spacing), "min_volume": float(min_volume)},
    )
    return make_tet_mesh(vertices, tets.reshape(-1, 4), metadata=metadata)


def tet_mesh_edges(mesh: TetMesh) -> np.ndarray:
    """Unique mesh edges as ``[E, 2]``, listing both directions."""

    pairs = mesh.tetrahedra[:, [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]].reshape(-1, 2)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    return np.concatenate([pairs, pairs[:, ::-1]])


def tet_mesh_boundary_faces(mesh: TetMesh) -> np.ndarray:
    """Triangles that belong to exactly one tetrahedron (the mesh surface), as ``[F, 3]``."""

    faces = mesh.tetrahedra[:, [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]]].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    _, index, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
    return faces[index[counts == 1]]


def tet_mesh_to_surface_mesh(mesh: TetMesh) -> trimesh.Trimesh:
    return trimesh.Trimesh(vertices=mesh.vertices, faces=tet_mesh_boundary_faces(mesh), process=True)


def locate_points_in_tet_mesh(mesh: TetMesh, points: Any, *, candidates: int = 64, tolerance: float = 1e-4):
    """Tetrahedron and barycentric weights for interpolating vertex values at each point.

    Returns ``(tet_index [N], barycentric [N, 4], inside [N])`` such that
    ``(barycentric[:, :, None] * values[tetrahedra[tet_index]]).sum(1)``
    interpolates per-vertex ``values``. For a point outside every candidate
    tetrahedron (``inside`` is False) the weights select its nearest vertex.
    """

    points_array = np.asarray(points, dtype=np.float64)
    vertices = mesh.vertices.astype(np.float64)
    tets = mesh.tetrahedra
    origin = vertices[tets[:, 0]]
    basis = np.stack([vertices[tets[:, i]] - origin for i in (1, 2, 3)], axis=2)
    inverse = np.linalg.inv(basis)
    k = min(int(candidates), len(tets))
    _, nearest = cKDTree(vertices[tets].mean(axis=1)).query(points_array, k=k)
    nearest = nearest.reshape(len(points_array), k)
    local = np.einsum("nkij,nkj->nki", inverse[nearest], points_array[:, None] - origin[nearest])
    weights = np.concatenate([1.0 - local.sum(-1, keepdims=True), local], axis=-1)
    contains = np.all(weights >= -tolerance, axis=-1)
    inside = contains.any(axis=1)
    rows = np.arange(len(points_array))
    choice = contains.argmax(axis=1)
    tet_index = nearest[rows, choice]
    barycentric = np.clip(weights[rows, choice], 0.0, 1.0)
    if (~inside).any():
        _, nearest_vertex = cKDTree(vertices).query(points_array[~inside])
        first_tet = np.empty(len(vertices), dtype=np.int64)
        first_tet[tets[::-1].reshape(-1)] = np.repeat(np.arange(len(tets))[::-1], 4)
        tet_index[~inside] = first_tet[nearest_vertex]
        barycentric[~inside] = tets[tet_index[~inside]] == nearest_vertex[:, None]
    barycentric = barycentric / barycentric.sum(axis=1, keepdims=True)
    return tet_index.astype(np.int64), barycentric.astype(np.float32), inside


def save_tet_mesh(
    mesh: TetMesh,
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Tetrahedral meshes can only be saved to .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, vertices=mesh.vertices, tetrahedra=mesh.tetrahedra)
    sidecar_metadata = metadata or mesh.metadata
    if sidecar_metadata is not None and write_sidecar:
        write_metadata(sidecar_metadata, metadata_sidecar_path(path))
    return path


def load_tet_mesh(path: str | Path) -> TetMesh:
    with np.load(Path(path), allow_pickle=False) as data:
        return make_tet_mesh(data["vertices"], data["tetrahedra"])


def tet_mesh_summary(mesh: TetMesh) -> dict[str, Any]:
    volumes = tet_mesh_volumes(mesh)
    return {
        "vertices": int(len(mesh.vertices)),
        "tetrahedra": int(len(mesh.tetrahedra)),
        "total_volume": float(np.abs(volumes).sum()),
        "min_volume": float(np.abs(volumes).min()) if len(volumes) else 0.0,
    }
