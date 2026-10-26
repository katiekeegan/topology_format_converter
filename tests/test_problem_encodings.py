import numpy as np
import pytest

from topology_format_converter import (
    PROBLEM_CHANNEL_NAMES,
    dense_channels_to_sparse,
    density_to_mesh,
    density_to_tet_mesh,
    gaussian_smooth_channels,
    load_attributed_points,
    load_graph_samples,
    load_sparse_voxels,
    load_tet_mesh,
    locate_points_in_tet_mesh,
    make_attributed_points,
    points_to_knn_graph,
    problem_channels,
    sample_grid_channels,
    sample_query_points,
    save_attributed_points,
    save_graph_samples,
    save_sparse_voxels,
    save_tet_mesh,
    scatter_grid_channels_to_points,
    sparse_to_dense,
    sparse_voxels_to_graph,
    subsample_attributed_points,
    tet_mesh_boundary_faces,
    tet_mesh_edges,
    tet_mesh_to_surface_mesh,
    tet_mesh_volumes,
    voxels_to_attributed_points,
)


def disc_problem():
    """A small disc-shaped design domain with two point loads and a supported rim segment."""
    shape = (12, 12, 3)
    i, j = np.meshgrid(np.arange(12), np.arange(12), indexing="ij")
    disc = ((i - 5.5) ** 2 + (j - 5.5) ** 2 <= 30)[:, :, None].repeat(3, axis=2)
    design = np.where(disc, -1.0, 0.0).astype(np.float32)
    dirichlet = np.zeros((3, *shape), dtype=np.float32)
    dirichlet[:, 0:2, 4:8, :] = disc[0:2, 4:8, :]
    force = np.zeros((3, *shape), dtype=np.float32)
    force[2, 6, 6, 1] = -5.0
    force[0, 9, 5, 1] = 2.0
    design[6, 6, 1] = design[9, 5, 1] = 1.0
    return problem_channels(dirichlet, design, force), disc


def test_problem_channels_order_and_names():
    channels, _ = disc_problem()
    assert channels.shape == (7, 12, 12, 3)
    assert len(PROBLEM_CHANNEL_NAMES) == 7 and PROBLEM_CHANNEL_NAMES[3] == "design_space"
    assert channels[3, 6, 6, 1] == 1.0 and channels[6, 6, 6, 1] == -5.0


def test_dense_channels_to_sparse_roundtrip(tmp_path):
    channels, disc = disc_problem()
    sparse = dense_channels_to_sparse(channels, mask=disc, channel_names=PROBLEM_CHANNEL_NAMES)
    assert sparse.values.shape == (disc.sum(), 7)
    np.testing.assert_array_equal(sparse_to_dense(sparse), channels * disc)
    path = save_sparse_voxels(sparse, tmp_path / "problem.npz")
    loaded = load_sparse_voxels(path)
    assert loaded.channel_names == PROBLEM_CHANNEL_NAMES
    np.testing.assert_array_equal(loaded.values, sparse.values)


def test_voxels_to_attributed_points_is_lossless(tmp_path):
    channels, disc = disc_problem()
    samples = voxels_to_attributed_points(channels, disc, channel_names=PROBLEM_CHANNEL_NAMES)
    assert samples.points.shape == (disc.sum(), 3)
    np.testing.assert_array_equal(sample_grid_channels(channels, samples.points, order=0), samples.features)
    loaded = load_attributed_points(save_attributed_points(samples, tmp_path / "points.npz"))
    assert loaded.channel_names == PROBLEM_CHANNEL_NAMES


def test_scatter_conserves_loads_and_targets_nearest_point():
    channels, _ = disc_problem()
    points = np.array([[6.2, 6.1, 1.0], [9.0, 5.0, 1.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    moved = scatter_grid_channels_to_points(channels[4:], points)
    np.testing.assert_allclose(moved.sum(axis=0), channels[4:].sum(axis=(1, 2, 3)))
    assert moved[0, 2] == -5.0 and moved[1, 0] == 2.0 and not moved[2].any()


def test_subsample_keeps_flagged_points():
    samples = make_attributed_points(np.random.rand(50, 3), np.arange(50.0))
    keep = np.zeros(50, dtype=bool)
    keep[[3, 17]] = True
    subset = subsample_attributed_points(samples, 10, keep=keep, seed=0)
    assert len(subset.points) == 10 and {3.0, 17.0} <= set(subset.features[:, 0])
    with pytest.raises(ValueError):
        subsample_attributed_points(samples, 1, keep=keep)


@pytest.mark.parametrize("connectivity, degree", [(6, 3), (18, 6), (26, 7)])
def test_sparse_voxels_to_graph_lattice_degrees(connectivity, degree, tmp_path):
    block = np.zeros((4, 4, 4), dtype=np.float32)
    block[1:3, 1:3, 1:3] = 1.0
    graph = sparse_voxels_to_graph(dense_channels_to_sparse(block), connectivity=connectivity)
    assert len(graph.points) == 8
    assert np.all(np.bincount(graph.edges[:, 0]) == degree)
    assert {tuple(edge) for edge in graph.edges} == {tuple(edge[::-1]) for edge in graph.edges}
    np.testing.assert_allclose(graph.edge_vectors, graph.points[graph.edges[:, 1]] - graph.points[graph.edges[:, 0]])
    loaded = load_graph_samples(save_graph_samples(graph, tmp_path / "graph.npz"))
    np.testing.assert_array_equal(loaded.edges, graph.edges)


def test_points_to_knn_graph_is_symmetric():
    graph = points_to_knn_graph(np.random.default_rng(0).random((30, 3)), np.ones(30), k=4)
    assert {tuple(edge) for edge in graph.edges} == {tuple(edge[::-1]) for edge in graph.edges}
    assert np.bincount(graph.edges[:, 0]).min() >= 4


def test_density_to_tet_mesh_fills_the_solid(tmp_path):
    channels, disc = disc_problem()
    loads = np.argwhere(np.any(channels[4:] != 0, axis=0))
    mesh = density_to_tet_mesh(disc.astype(np.float32), interior_spacing=2, extra_points=loads)
    volumes = tet_mesh_volumes(mesh)
    assert np.all(volumes > 0)
    surface = density_to_mesh(np.pad(disc.astype(np.float32), 1), threshold=0.5)
    assert abs(volumes.sum() - abs(surface.volume)) / abs(surface.volume) < 0.01
    assert np.array_equal(np.unique(mesh.tetrahedra), np.arange(len(mesh.vertices)))
    vertex_set = {tuple(np.round(vertex, 4)) for vertex in mesh.vertices}
    assert all(tuple(load.astype(np.float32)) in vertex_set for load in loads)
    assert len(tet_mesh_boundary_faces(mesh)) > 0 and len(tet_mesh_edges(mesh)) > 0
    assert len(tet_mesh_to_surface_mesh(mesh).faces) > 0
    loaded = load_tet_mesh(save_tet_mesh(mesh, tmp_path / "mesh.npz"))
    np.testing.assert_array_equal(loaded.tetrahedra, mesh.tetrahedra)


def test_barycentric_location_interpolates_linear_fields():
    _, disc = disc_problem()
    mesh = density_to_tet_mesh(disc.astype(np.float32), interior_spacing=2)
    linear = mesh.vertices @ np.array([1.0, 2.0, -1.0]) + 0.5
    queries = np.argwhere(disc).astype(np.float32)
    tet, weights, inside = locate_points_in_tet_mesh(mesh, queries)
    interpolated = (weights * linear[mesh.tetrahedra[tet]]).sum(axis=1)
    np.testing.assert_allclose(interpolated[inside], (queries @ np.array([1.0, 2.0, -1.0]) + 0.5)[inside], atol=1e-3)
    assert inside.mean() > 0.9
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-5)


def test_sample_query_points_bounds_and_concentration():
    near = np.zeros((1, 3), dtype=np.float32)
    points = sample_query_points((10, 10, 10), 400, near_points=near, near_fraction=0.5, noise_std=0.01, seed=0)
    assert points.shape == (400, 3) and points.min() >= -1 and points.max() <= 1
    assert (np.linalg.norm(points, axis=1) < 0.05).sum() >= 190


def test_gaussian_smooth_spreads_point_loads():
    grid = np.zeros((1, 15, 15, 15), dtype=np.float32)
    grid[0, 7, 7, 7] = 1.0
    smooth = gaussian_smooth_channels(grid, sigma=1.5)
    assert abs(smooth.sum() - 1.0) < 1e-4 and smooth[0, 7, 7, 8] > 0
