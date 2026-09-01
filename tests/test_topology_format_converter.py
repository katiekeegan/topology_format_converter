import numpy as np
import pytest

from topology_format_converter import (
    cache_to_mesh,
    convert_file,
    density_to_mesh,
    density_to_training_cache,
    dense_to_sparse,
    export_mesh,
    field_samples_summary,
    load_training_cache,
    load_volume,
    load_field_samples,
    load_pointcloud,
    load_sdf_samples,
    load_sparse_voxels,
    make_field_samples,
    make_sparse_voxels,
    make_sdf_samples,
    pointcloud_to_distance_grid,
    pointcloud_to_occupancy,
    save_field_samples,
    save_sdf_samples,
    save_sparse_voxels,
    save_signed_distance,
    save_training_cache,
    save_volume,
    save_volume_vtk,
    signed_distance_from_density,
    sparse_to_dense,
    truncate_distance_grid,
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


def test_truncate_distance_grid():
    grid = np.array(
        [[[-2.0, -0.5], [0.2, 3.0]], [[-1.0, 0.0], [0.5, 2.0]]],
        dtype=np.float32,
    )
    truncated = truncate_distance_grid(grid, truncation=0.75)
    assert float(truncated.min()) == -0.75
    assert float(truncated.max()) == 0.75


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

    result = cli_main(
        [
            "mesh-to-pointcloud",
            str(mesh_path),
            "--out",
            str(points_path),
            "--num-points",
            "20",
            "--seed",
            "123",
        ]
    )
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


def test_sdf_samples_roundtrip_npz_and_csv(tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, -0.5]], dtype=np.float32)
    sdf = np.array([-0.1, 0.2], dtype=np.float32)
    normals = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    samples = make_sdf_samples(points, sdf, normals=normals)

    npz_path = tmp_path / "samples.npz"
    csv_path = tmp_path / "samples.csv"
    save_sdf_samples(samples, npz_path)
    save_sdf_samples(samples, csv_path)

    loaded_npz = load_sdf_samples(npz_path)
    loaded_csv = load_sdf_samples(csv_path)
    np.testing.assert_allclose(loaded_npz.points, points)
    np.testing.assert_allclose(loaded_npz.sdf, sdf)
    np.testing.assert_allclose(loaded_csv.points, points)
    np.testing.assert_allclose(loaded_csv.sdf, sdf)


def test_field_samples_roundtrip_npz_and_csv(tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, -0.5]], dtype=np.float32)
    values = np.array([0.1, 0.9], dtype=np.float32)
    samples = make_field_samples(points, values, field_name="stress")

    npz_path = tmp_path / "field.npz"
    csv_path = tmp_path / "field.csv"
    save_field_samples(samples, npz_path)
    save_field_samples(samples, csv_path)

    loaded_npz = load_field_samples(npz_path)
    loaded_csv = load_field_samples(csv_path, values_key="stress")
    np.testing.assert_allclose(loaded_npz.points, points)
    np.testing.assert_allclose(loaded_npz.values, values)
    assert loaded_npz.field_name == "stress"
    assert field_samples_summary(loaded_npz)["field_name"] == "stress"
    np.testing.assert_allclose(loaded_csv.values, values)


def test_sparse_voxels_roundtrip(tmp_path):
    sparse = dense_to_sparse(cube_density(), threshold=0.5)
    sparse_path = tmp_path / "sparse.npz"
    save_sparse_voxels(sparse, sparse_path)

    loaded = load_sparse_voxels(sparse_path)
    dense = sparse_to_dense(loaded)
    assert loaded.indices.shape[1] == 3
    assert loaded.values.shape == (64,)
    np.testing.assert_allclose(dense, cube_density())

    manual = make_sparse_voxels([[0, 0, 0]], [2.0], shape=(2, 2, 2))
    assert sparse_to_dense(manual)[0, 0, 0] == 2.0


def test_pointcloud_to_occupancy_and_distance_grid():
    points = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    occupancy = pointcloud_to_occupancy(
        points, shape=(4, 4, 4), coordinate_mode="unit-box"
    )
    assert occupancy[0, 0, 0] == 1.0
    assert occupancy[3, 3, 3] == 1.0
    assert occupancy.sum() == 2.0

    distances = pointcloud_to_distance_grid(
        points, shape=(4, 4, 4), coordinate_mode="unit-box"
    )
    assert distances.shape == (4, 4, 4)
    assert distances[0, 0, 0] == 0.0
    assert distances[3, 3, 3] == 0.0


def test_general_convert_file_volume_to_mesh_and_pointcloud_to_sdf(tmp_path):
    density_path = tmp_path / "density.npz"
    mesh_path = tmp_path / "mesh.obj"
    points_path = tmp_path / "points.npz"
    sdf_path = tmp_path / "sdf_grid.npz"

    save_volume(density_path, cube_density())
    convert_file(
        density_path,
        mesh_path,
        source_modality="volume",
        target_modality="mesh",
        coordinate_mode="unit-box",
    )
    assert mesh_path.exists()

    result = cli_main(
        [
            "convert-file",
            str(mesh_path),
            "--out",
            str(points_path),
            "--target-modality",
            "pointcloud",
            "--num-points",
            "24",
            "--seed",
            "123",
        ]
    )
    assert result == 0
    points, normals = load_pointcloud(points_path)
    assert points.shape == (24, 3)
    assert normals is not None

    with pytest.warns(UserWarning, match="unsigned nearest-surface distance grid"):
        result = cli_main(
            [
                "convert-file",
                str(points_path),
                "--out",
                str(sdf_path),
                "--source-modality",
                "pointcloud",
                "--target-modality",
                "sdf-grid",
                "--resolution",
                "6",
            ]
        )
    assert result == 0
    distance = load_volume(sdf_path, key="sdf")
    assert distance.shape == (6, 6, 6)
    assert float(distance.min()) >= 0.0


def test_general_convert_file_distance_grid_targets(tmp_path):
    density_path = tmp_path / "density.npz"
    udf_path = tmp_path / "udf.npz"
    tsdf_path = tmp_path / "tsdf.npz"
    save_volume(density_path, cube_density())

    convert_file(
        density_path, udf_path, source_modality="volume", target_modality="udf-grid"
    )
    convert_file(
        density_path,
        tsdf_path,
        source_modality="volume",
        target_modality="tsdf-grid",
        truncation=1.25,
    )

    udf = load_volume(udf_path, key="udf")
    tsdf = load_volume(tsdf_path, key="tsdf")
    assert float(udf.min()) >= 0.0
    assert float(tsdf.min()) >= -1.25
    assert float(tsdf.max()) <= 1.25


def test_general_convert_file_sparse_voxels_to_volume_and_pointcloud(tmp_path):
    density_path = tmp_path / "density.npz"
    sparse_path = tmp_path / "sparse.npz"
    roundtrip_path = tmp_path / "roundtrip.npz"
    points_path = tmp_path / "points.npz"
    save_volume(density_path, cube_density())

    convert_file(
        density_path,
        sparse_path,
        source_modality="occupancy",
        target_modality="sparse-voxels",
    )
    convert_file(
        sparse_path,
        roundtrip_path,
        source_modality="sparse-voxels",
        target_modality="occupancy",
    )
    convert_file(
        sparse_path,
        points_path,
        source_modality="sparse-voxels",
        target_modality="pointcloud",
        coordinate_mode="voxel",
    )

    loaded = load_sparse_voxels(sparse_path)
    occupancy = load_volume(roundtrip_path, key="occupancy")
    points, normals = load_pointcloud(points_path)
    assert loaded.indices.shape == (64, 3)
    np.testing.assert_allclose(occupancy, cube_density())
    assert points.shape == (64, 3)
    assert normals is None


def test_general_convert_file_field_samples_and_sdf_samples(tmp_path):
    points = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]], dtype=np.float32)
    values = np.array([-0.2, 0.3], dtype=np.float32)
    field_path = tmp_path / "field.npz"
    sdf_path = tmp_path / "samples.npz"
    field_roundtrip = tmp_path / "field_roundtrip.csv"
    save_field_samples(make_field_samples(points, values, field_name="sdf"), field_path)

    convert_file(
        field_path,
        sdf_path,
        source_modality="field-samples",
        target_modality="sdf-samples",
    )
    convert_file(
        sdf_path,
        field_roundtrip,
        source_modality="sdf-samples",
        target_modality="field-samples",
    )

    sdf_samples = load_sdf_samples(sdf_path)
    loaded_field = load_field_samples(field_roundtrip, values_key="sdf")
    np.testing.assert_allclose(sdf_samples.sdf, values)
    np.testing.assert_allclose(loaded_field.values, values)


def test_general_convert_file_mesh_to_surface_occupancy(tmp_path):
    mesh_path = tmp_path / "cube.obj"
    occupancy_path = tmp_path / "occupancy.npz"
    mesh = density_to_mesh(cube_density(), threshold=0.5, coordinate_mode="unit-box")
    export_mesh(mesh, mesh_path)

    convert_file(
        mesh_path,
        occupancy_path,
        source_modality="mesh",
        target_modality="occupancy",
        coordinate_mode="unit-box",
        resolution=8,
        num_points=200,
        mark_radius=1,
        seed=123,
    )

    occupancy = load_volume(occupancy_path, key="occupancy")
    assert occupancy.shape == (8, 8, 8)
    assert occupancy.max() == 1.0
    assert occupancy.sum() > 0.0


def test_general_convert_file_pointcloud_to_sdf_warns_and_marks_unsigned(tmp_path):
    points_path = tmp_path / "points.npz"
    sdf_path = tmp_path / "sdf_grid.npz"
    np.savez_compressed(
        points_path, points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    )

    with pytest.warns(UserWarning, match="unsigned nearest-surface distance grid"):
        convert_file(
            points_path,
            sdf_path,
            source_modality="pointcloud",
            target_modality="sdf-grid",
            resolution=4,
        )

    metadata = sdf_path.with_suffix(".npz.json").read_text()
    assert '"sdf_sign": "unsigned"' in metadata
    assert "unsigned_nearest_surface" in metadata


def test_general_convert_file_pointcloud_to_udf_and_rejects_tsdf(tmp_path):
    points_path = tmp_path / "points.npz"
    udf_path = tmp_path / "udf.npz"
    tsdf_path = tmp_path / "tsdf.npz"
    np.savez_compressed(
        points_path, points=np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    )

    convert_file(
        points_path,
        udf_path,
        source_modality="pointcloud",
        target_modality="udf-grid",
        resolution=4,
    )
    udf = load_volume(udf_path, key="udf")
    assert udf.shape == (4, 4, 4)
    assert float(udf.min()) >= 0.0

    with pytest.raises(ValueError, match="sign information is unavailable"):
        convert_file(
            points_path,
            tsdf_path,
            source_modality="pointcloud",
            target_modality="tsdf-grid",
            resolution=4,
        )
