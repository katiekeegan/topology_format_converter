import numpy as np

from gto_format_converter import (
    density_to_mesh,
    density_to_training_cache,
    load_training_cache,
    save_training_cache,
    signed_distance_from_density,
)


def cube_density():
    density = np.zeros((8, 8, 8), dtype=np.float32)
    density[2:6, 2:6, 2:6] = 1.0
    return density


def test_signed_distance_negative_inside():
    sdf = signed_distance_from_density(cube_density(), threshold=0.5)
    assert sdf[3, 3, 3] < 0
    assert sdf[0, 0, 0] > 0


def test_density_to_mesh_returns_faces():
    mesh = density_to_mesh(cube_density(), threshold=0.5, coordinate_mode="training")
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0


def test_training_cache_roundtrip(tmp_path):
    cache = density_to_training_cache(
        cube_density(),
        threshold=0.5,
        num_surface_points=32,
        num_query_points=64,
        seed=123,
    )
    out = tmp_path / "cache.npz"
    save_training_cache(cache, out)
    loaded = load_training_cache(out)
    assert loaded.density.shape == (8, 8, 8)
    assert loaded.surface_points.shape == (32, 3)
    assert loaded.query_points.shape == (64, 3)
    assert loaded.query_sdf.shape == (64,)
