from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from .attributes import _channel_names
from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata
from .sparse import SparseVoxels, make_sparse_voxels
from .volume import voxel_indices_to_coords

VALID_CONNECTIVITIES = (6, 18, 26)


@dataclass
class GraphSamples:
    """A graph with positioned nodes, a feature vector per node, and directed edges.

    ``edges`` is ``[E, 2]`` (source, target) and lists both directions of every
    undirected edge. ``edge_vectors`` is ``points[target] - points[source]``.
    """

    points: np.ndarray
    features: np.ndarray
    edges: np.ndarray
    edge_vectors: np.ndarray
    channel_names: tuple[str, ...]
    metadata: Optional[ConversionMetadata] = None


def make_graph_samples(
    points: Any,
    features: Any,
    edges: Any,
    *,
    channel_names: Optional[Sequence[str]] = None,
    metadata: Optional[ConversionMetadata] = None,
) -> GraphSamples:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3]; got {points_array.shape}.")
    features_array = np.asarray(features, dtype=np.float32)
    if features_array.ndim == 1:
        features_array = features_array[:, None]
    if features_array.ndim != 2 or features_array.shape[0] != len(points_array):
        raise ValueError(f"features must have shape [{len(points_array)}, C]; got {features_array.shape}.")
    edges_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if len(edges_array) and (edges_array.min() < 0 or edges_array.max() >= len(points_array)):
        raise ValueError("edges reference nodes outside the graph.")
    return GraphSamples(
        points=points_array,
        features=features_array,
        edges=edges_array,
        edge_vectors=(points_array[edges_array[:, 1]] - points_array[edges_array[:, 0]]).astype(np.float32),
        channel_names=_channel_names(channel_names, features_array.shape[1]),
        metadata=metadata,
    )


def _lattice_offsets(connectivity: int) -> np.ndarray:
    if connectivity not in VALID_CONNECTIVITIES:
        raise ValueError(f"connectivity must be one of {VALID_CONNECTIVITIES}; got {connectivity}.")
    offsets = np.array([offset for offset in product((-1, 0, 1), repeat=3) if any(offset)], dtype=np.int64)
    steps = np.abs(offsets).sum(axis=1)
    return offsets[steps <= {6: 1, 18: 2, 26: 3}[connectivity]]


def sparse_voxels_to_graph(
    sparse: SparseVoxels,
    *,
    connectivity: int = 26,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
) -> GraphSamples:
    """Graph over the active sites of sparse voxels, linking lattice neighbours.

    Nodes sit at voxel centres and carry the sites' values as features; an edge
    joins two active sites that are neighbours under 6-, 18- or 26-connectivity.
    """

    sparse = make_sparse_voxels(sparse.indices, sparse.values, shape=sparse.shape, channel_names=sparse.channel_names)
    lookup = np.full(sparse.shape, -1, dtype=np.int64)
    lookup[tuple(sparse.indices.T)] = np.arange(len(sparse.indices))
    shape = np.asarray(sparse.shape)
    edges = []
    for offset in _lattice_offsets(connectivity):
        neighbours = sparse.indices + offset
        inside = np.all((neighbours >= 0) & (neighbours < shape), axis=1)
        target = np.full(len(sparse.indices), -1, dtype=np.int64)
        target[inside] = lookup[tuple(neighbours[inside].T)]
        valid = target >= 0
        edges.append(np.stack([np.flatnonzero(valid), target[valid]], axis=1))
    points = voxel_indices_to_coords(
        sparse.indices.astype(np.float32), sparse.shape, coordinate_mode=coordinate_mode, spacing=spacing
    )
    return make_graph_samples(
        points,
        sparse.values,
        np.concatenate(edges) if edges else np.empty((0, 2), dtype=np.int64),
        channel_names=sparse.channel_names,
    )


def points_to_knn_graph(
    points: Any,
    features: Any,
    *,
    k: int = 8,
    channel_names: Optional[Sequence[str]] = None,
) -> GraphSamples:
    """Graph linking every point to its ``k`` nearest neighbours (symmetrized)."""

    points_array = np.asarray(points, dtype=np.float32)
    if k <= 0 or k >= len(points_array):
        raise ValueError(f"k must be in [1, {len(points_array) - 1}]; got {k}.")
    _, neighbours = cKDTree(points_array).query(points_array, k=k + 1)
    source = np.repeat(np.arange(len(points_array)), k)
    target = neighbours[:, 1:].reshape(-1)
    pairs = np.concatenate([np.stack([source, target], 1), np.stack([target, source], 1)])
    return make_graph_samples(points_array, features, np.unique(pairs, axis=0), channel_names=channel_names)


def save_graph_samples(
    graph: GraphSamples,
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Graphs can only be saved to .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        points=graph.points,
        features=graph.features,
        edges=graph.edges,
        channel_names=np.asarray(graph.channel_names),
    )
    sidecar_metadata = metadata or graph.metadata
    if sidecar_metadata is not None and write_sidecar:
        write_metadata(sidecar_metadata, metadata_sidecar_path(path))
    return path


def load_graph_samples(path: str | Path) -> GraphSamples:
    with np.load(Path(path), allow_pickle=False) as data:
        names = [str(name) for name in data["channel_names"]] if "channel_names" in data else None
        return make_graph_samples(data["points"], data["features"], data["edges"], channel_names=names)


def graph_samples_summary(graph: GraphSamples) -> dict[str, Any]:
    degree = np.bincount(graph.edges[:, 0], minlength=len(graph.points)) if len(graph.edges) else np.zeros(len(graph.points))
    return {
        "nodes": int(len(graph.points)),
        "edges": int(len(graph.edges)),
        "channels": list(graph.channel_names),
        "mean_degree": float(degree.mean()) if len(degree) else 0.0,
    }
