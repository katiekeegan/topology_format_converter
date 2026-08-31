import numpy as np

from topology_format_converter import (
    cache_to_mesh,
    density_to_mesh,
    density_to_training_cache,
    export_mesh,
    load_training_cache,
    load_volume,
    save_signed_distance,
    save_training_cache,
    save_volume,
    save_volume_vtk,
    signed_distance_from_density,
)
from topology_format_converter.cli import main as cli_main


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


def test_save_volume_and_sdf(tmp_path):
    density_path = tmp_path / "density.npy"
    sdf_path = tmp_path / "sdf.npz"
    save_volume(density_path, cube_density())
    save_signed_distance(sdf_path, load_volume(density_path), threshold=0.5)

    loaded_sdf = load_volume(sdf_path, key="sdf")
    assert loaded_sdf.shape == (8, 8, 8)
    assert loaded_sdf[3, 3, 3] < 0


def test_save_volume_vtk(tmp_path):
    vtk_path = tmp_path / "density.vtk"
    save_volume_vtk(vtk_path, cube_density(), scalar_name="density")
    text = vtk_path.read_text()
    assert "DATASET STRUCTURED_POINTS" in text
    assert "DIMENSIONS 8 8 8" in text
    assert "SCALARS density float 1" in text


def test_cache_to_mesh(tmp_path):
    cache = density_to_training_cache(
        cube_density(),
        threshold=0.5,
        num_surface_points=16,
        num_query_points=32,
        seed=123,
    )
    mesh = cache_to_mesh(cache)
    assert len(mesh.faces) > 0


def test_mesh_to_pointcloud_cli(tmp_path):
    mesh_path = tmp_path / "cube.obj"
    points_path = tmp_path / "points.npz"
    mesh = density_to_mesh(cube_density(), threshold=0.5)
    export_mesh(mesh, mesh_path)

    result = cli_main([
        "mesh-to-pointcloud",
        str(mesh_path),
        "--out",
        str(points_path),
        "--num-points",
        "20",
        "--seed",
        "123",
    ])
    assert result == 0
    with np.load(points_path) as data:
        assert data["points"].shape == (20, 3)
        assert data["normals"].shape == (20, 3)


def test_volume_to_sdf_cli(tmp_path):
    density_path = tmp_path / "density.npy"
    sdf_path = tmp_path / "sdf.npy"
    np.save(density_path, cube_density())
    result = cli_main(["volume-to-sdf", str(density_path), "--out", str(sdf_path)])
    assert result == 0
    assert sdf_path.with_suffix(".npy.json").exists()
    assert load_volume(sdf_path).shape == (8, 8, 8)
