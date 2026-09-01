from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates

from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata

VALID_COORDINATE_MODES = ("voxel", "unit-box", "training")


def _torch_module():
    try:
        import torch

        return torch
    except ImportError:
        return None


def to_numpy(value: Any) -> np.ndarray:
    """Convert tensors/arrays/scalars to a detached CPU numpy array."""

    torch = _torch_module()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def as_3d_array(value: Any, *, name: str = "volume") -> np.ndarray:
    array = np.asarray(to_numpy(value)).squeeze()
    if array.ndim != 3:
        raise ValueError(f"{name} must be 3-D after squeeze; got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


def parse_spacing(
    spacing: Optional[Iterable[float]],
) -> Optional[tuple[float, float, float]]:
    if spacing is None:
        return None
    values = tuple(float(value) for value in spacing)
    if len(values) != 3:
        raise ValueError(f"spacing must contain exactly 3 values; got {values}.")
    if any(value <= 0 for value in values):
        raise ValueError(f"spacing values must be positive; got {values}.")
    return values


def validate_coordinate_mode(coordinate_mode: str) -> str:
    if coordinate_mode not in VALID_COORDINATE_MODES:
        raise ValueError(
            f"coordinate_mode must be one of {VALID_COORDINATE_MODES}; got {coordinate_mode!r}."
        )
    return coordinate_mode


def extract_density(obj: Any, key: Optional[str] = None) -> np.ndarray:
    """Extract a density volume from common arrays, dicts, tensors, and DL4TO tuples."""

    if key is not None:
        if not isinstance(obj, dict):
            raise ValueError(
                "--input-key can only be used when the loaded object is a dict-like object."
            )
        if key not in obj:
            raise KeyError(
                f"Could not find key {key!r}. Available keys: {sorted(obj.keys())}"
            )
        return as_3d_array(obj[key], name=key)

    if isinstance(obj, dict):
        for candidate in (
            "density",
            "theta",
            "volume",
            "voxel_grid",
            "occupancy",
            "sdf",
            "sdf_grid",
            "udf",
            "udf_grid",
            "tsdf",
            "tsdf_grid",
        ):
            if candidate in obj:
                return as_3d_array(obj[candidate], name=candidate)
        raise KeyError(
            f"Could not infer a density key. Available keys: {sorted(obj.keys())}"
        )

    if isinstance(obj, (tuple, list)) and len(obj) >= 2:
        solution = obj[1]
        if hasattr(solution, "\u03b8"):
            return as_3d_array(getattr(solution, "\u03b8"), name="solution.theta")

    if hasattr(obj, "\u03b8"):
        return as_3d_array(getattr(obj, "\u03b8"), name="solution.theta")

    return as_3d_array(obj)


def load_volume(path: str | Path, key: Optional[str] = None) -> np.ndarray:
    """Load a density/SDF/volume array from .npy, .npz, .pt, or .pth."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return as_3d_array(np.load(path), name=str(path))

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if key is None:
                for candidate in (
                    "density",
                    "theta",
                    "volume",
                    "voxel_grid",
                    "occupancy",
                    "sdf",
                    "sdf_grid",
                    "udf",
                    "udf_grid",
                    "tsdf",
                    "tsdf_grid",
                ):
                    if candidate in data:
                        key = candidate
                        break
                if key is None:
                    keys = [item for item in data.files if item != "metadata_json"]
                    if not keys:
                        raise KeyError(f"No array keys found in {path}.")
                    key = keys[0]
            if key not in data:
                raise KeyError(
                    f"Could not find key {key!r}. Available keys: {data.files}"
                )
            return as_3d_array(data[key], name=key)

    if suffix in {".pt", ".pth"}:
        torch = _torch_module()
        if torch is None:
            raise ImportError("torch is required to load .pt/.pth volume files.")
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
        return extract_density(obj, key=key)

    raise ValueError(
        f"Unsupported volume file extension {suffix!r}. Use .npy, .npz, .pt, or .pth."
    )


def save_volume_npz(path: str | Path, volume: Any, *, key: str = "density") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: as_3d_array(volume)})
    return path


def save_volume(
    path: str | Path,
    volume: Any,
    *,
    key: str = "density",
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    """Save a 3-D volume as .npy or .npz."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = as_3d_array(volume, name=key)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        np.save(path, array)
    elif suffix == ".npz":
        np.savez_compressed(path, **{key: array})
    else:
        raise ValueError(
            f"Unsupported volume output extension {suffix!r}. Use .npy or .npz."
        )
    if metadata is not None and write_sidecar:
        write_metadata(metadata, metadata_sidecar_path(path))
    return path


def save_volume_arrays_npz(
    path: str | Path,
    arrays: dict[str, Any],
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    """Save multiple arrays in one compressed .npz archive."""

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Multiple arrays can only be saved to .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, **{key: np.asarray(to_numpy(value)) for key, value in arrays.items()}
    )
    if metadata is not None and write_sidecar:
        write_metadata(metadata, metadata_sidecar_path(path))
    return path


def threshold_density(density: Any, threshold: float = 0.5) -> np.ndarray:
    return as_3d_array(density, name="density") >= float(threshold)


def signed_distance_from_density(
    density: Any,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
) -> np.ndarray:
    """Return signed distance from a density grid.

    The sign convention matches the training code: values are negative inside
    solid material and positive outside.
    """

    mask = threshold_density(density, threshold=threshold)
    sampling = parse_spacing(spacing)
    outside = distance_transform_edt(~mask, sampling=sampling)
    inside = distance_transform_edt(mask, sampling=sampling)
    return (outside - inside).astype(np.float32, copy=False)


def truncate_distance_grid(distance_grid: Any, truncation: float) -> np.ndarray:
    """Clip a signed-distance-like grid to a truncation band."""

    truncation_value = float(truncation)
    if truncation_value <= 0:
        raise ValueError(f"truncation must be positive; got {truncation}.")
    distance = as_3d_array(distance_grid, name="distance_grid")
    return np.clip(distance, -truncation_value, truncation_value).astype(
        np.float32, copy=False
    )


def save_signed_distance(
    path: str | Path,
    density: Any,
    *,
    threshold: float = 0.5,
    spacing: Optional[Iterable[float]] = None,
    normalize: bool = False,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    density_array = as_3d_array(density, name="density")
    sdf = signed_distance_from_density(
        density_array, threshold=threshold, spacing=spacing
    )
    if normalize:
        scale = sdf_normalization_scale(density_array.shape, spacing=spacing)
        if scale > 0:
            sdf = sdf / scale
    return save_volume(
        path, sdf, key="sdf", metadata=metadata, write_sidecar=write_sidecar
    )


def save_volume_vtk(
    path: str | Path,
    volume: Any,
    *,
    scalar_name: str = "density",
    spacing: Optional[Iterable[float]] = None,
    origin: Iterable[float] = (0.0, 0.0, 0.0),
) -> Path:
    """Write a legacy ASCII VTK structured-points file for ParaView."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = as_3d_array(volume, name=scalar_name)
    spacing_values = parse_spacing(spacing) or (1.0, 1.0, 1.0)
    origin_values = tuple(float(value) for value in origin)
    if len(origin_values) != 3:
        raise ValueError("origin must contain exactly 3 values.")

    values = array.astype(np.float32, copy=False).ravel(order="F")
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("Topology volume\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_POINTS\n")
        handle.write(f"DIMENSIONS {array.shape[0]} {array.shape[1]} {array.shape[2]}\n")
        handle.write(
            f"ORIGIN {origin_values[0]} {origin_values[1]} {origin_values[2]}\n"
        )
        handle.write(
            f"SPACING {spacing_values[0]} {spacing_values[1]} {spacing_values[2]}\n"
        )
        handle.write(f"POINT_DATA {array.size}\n")
        handle.write(f"SCALARS {scalar_name} float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for start in range(0, len(values), 9):
            handle.write(
                " ".join(f"{value:.9g}" for value in values[start : start + 9])
            )
            handle.write("\n")
    return path


def sdf_normalization_scale(
    shape: Iterable[int], spacing: Optional[Iterable[float]] = None
) -> float:
    shape_array = np.asarray(tuple(int(value) for value in shape), dtype=np.float32)
    half_extent = (shape_array - 1.0) / 2.0
    spacing_values = parse_spacing(spacing)
    if spacing_values is not None:
        half_extent = half_extent * np.asarray(spacing_values, dtype=np.float32)
    return float(np.linalg.norm(half_extent))


def voxel_indices_to_coords(
    points: Any,
    shape: Iterable[int],
    *,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
) -> np.ndarray:
    coordinate_mode = validate_coordinate_mode(coordinate_mode)
    points_array = np.asarray(points, dtype=np.float32)
    shape_array = np.asarray(tuple(int(value) for value in shape), dtype=np.float32)
    spacing_values = np.asarray(
        parse_spacing(spacing) or (1.0, 1.0, 1.0), dtype=np.float32
    )

    if coordinate_mode == "voxel":
        return points_array * spacing_values
    if coordinate_mode == "unit-box":
        denom = np.maximum(shape_array - 1.0, 1.0)
        return points_array / denom * 2.0 - 1.0

    scale = float(shape_array.max() / 2.0)
    return points_array / scale - 1.0


def coords_to_voxel_indices(
    points: Any,
    shape: Iterable[int],
    *,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
) -> np.ndarray:
    coordinate_mode = validate_coordinate_mode(coordinate_mode)
    points_array = np.asarray(points, dtype=np.float32)
    shape_array = np.asarray(tuple(int(value) for value in shape), dtype=np.float32)
    spacing_values = np.asarray(
        parse_spacing(spacing) or (1.0, 1.0, 1.0), dtype=np.float32
    )

    if coordinate_mode == "voxel":
        return points_array / spacing_values
    if coordinate_mode == "unit-box":
        return (points_array + 1.0) * 0.5 * np.maximum(shape_array - 1.0, 1.0)

    scale = float(shape_array.max() / 2.0)
    return (points_array + 1.0) * scale


def coordinate_bounds(
    shape: Iterable[int],
    *,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    coordinate_mode = validate_coordinate_mode(coordinate_mode)
    shape_array = np.asarray(tuple(int(value) for value in shape), dtype=np.float32)
    spacing_values = np.asarray(
        parse_spacing(spacing) or (1.0, 1.0, 1.0), dtype=np.float32
    )

    if coordinate_mode == "voxel":
        return np.zeros(3, dtype=np.float32), (shape_array - 1.0) * spacing_values
    return -np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)


def sample_sdf_grid(
    sdf_grid: Any,
    points: Any,
    *,
    coordinate_mode: str = "voxel",
    spacing: Optional[Iterable[float]] = None,
    order: int = 1,
    mode: str = "nearest",
) -> np.ndarray:
    sdf_array = as_3d_array(sdf_grid, name="sdf_grid")
    indices = coords_to_voxel_indices(
        points,
        sdf_array.shape,
        coordinate_mode=coordinate_mode,
        spacing=spacing,
    )
    values = map_coordinates(
        sdf_array,
        [indices[:, 0], indices[:, 1], indices[:, 2]],
        order=order,
        mode=mode,
    )
    return values.astype(np.float32, copy=False)
