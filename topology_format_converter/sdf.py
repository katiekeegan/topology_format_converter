from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata


@dataclass
class SDFSamples:
    """Point samples with signed-distance values."""

    points: np.ndarray
    sdf: np.ndarray
    normals: Optional[np.ndarray] = None
    metadata: Optional[ConversionMetadata] = None


def _validate_points(points: Any) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3]; got {points_array.shape}.")
    return points_array


def _validate_sdf(sdf: Any, count: int) -> np.ndarray:
    sdf_array = np.asarray(sdf, dtype=np.float32).reshape(-1)
    if sdf_array.shape != (count,):
        raise ValueError(f"sdf must have shape ({count},); got {sdf_array.shape}.")
    return sdf_array


def _validate_normals(normals: Any, points_shape: tuple[int, int]) -> np.ndarray:
    normals_array = np.asarray(normals, dtype=np.float32)
    if normals_array.shape != points_shape:
        raise ValueError(f"normals must have shape {points_shape}; got {normals_array.shape}.")
    return normals_array


def make_sdf_samples(
    points: Any,
    sdf: Any,
    *,
    normals: Optional[Any] = None,
    metadata: Optional[ConversionMetadata] = None,
) -> SDFSamples:
    points_array = _validate_points(points)
    sdf_array = _validate_sdf(sdf, len(points_array))
    normals_array = None if normals is None else _validate_normals(normals, points_array.shape)
    return SDFSamples(points=points_array, sdf=sdf_array, normals=normals_array, metadata=metadata)


def load_sdf_samples(
    path: str | Path,
    *,
    points_key: str = "query_points",
    sdf_key: str = "query_sdf",
    normals_key: Optional[str] = None,
) -> SDFSamples:
    """Load SDF point samples from .npz or .csv.

    Common alternate NPZ keys are accepted when the requested keys are absent:
    `points`, `coords`, or `xyz` for positions and `sdf`, `signed_distance`, or
    `values` for signed-distance values.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            point_key = points_key
            if point_key not in data:
                for candidate in ("points", "coords", "xyz", "surface_points", "query_points"):
                    if candidate in data:
                        point_key = candidate
                        break
            value_key = sdf_key
            if value_key not in data:
                for candidate in ("sdf", "signed_distance", "values", "query_sdf"):
                    if candidate in data:
                        value_key = candidate
                        break
            if point_key not in data:
                raise KeyError(f"Could not find point samples in {path}. Available keys: {data.files}")
            if value_key not in data:
                raise KeyError(f"Could not find SDF values in {path}. Available keys: {data.files}")
            normals = data[normals_key] if normals_key and normals_key in data else None
            return make_sdf_samples(data[point_key], data[value_key], normals=normals)

    if suffix == ".csv":
        try:
            named = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
            if named.dtype.names and {"x", "y", "z"}.issubset(named.dtype.names):
                points = np.column_stack([named["x"], named["y"], named["z"]])
                for candidate in ("sdf", "signed_distance", "value"):
                    if candidate in named.dtype.names:
                        sdf = named[candidate]
                        break
                else:
                    raise ValueError("CSV SDF samples need an sdf, signed_distance, or value column.")
                normals = None
                if {"nx", "ny", "nz"}.issubset(named.dtype.names):
                    normals = np.column_stack([named["nx"], named["ny"], named["nz"]])
                return make_sdf_samples(points, sdf, normals=normals)
        except ValueError:
            pass
        values = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] < 4:
            raise ValueError(f"CSV SDF samples must have at least 4 columns; got {values.shape[1]}.")
        normals = values[:, 4:7] if values.shape[1] >= 7 else None
        return make_sdf_samples(values[:, :3], values[:, 3], normals=normals)

    raise ValueError(f"Unsupported SDF sample extension {suffix!r}. Use .npz or .csv.")


def save_sdf_samples(
    samples: SDFSamples,
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = make_sdf_samples(samples.points, samples.sdf, normals=samples.normals)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        arrays = {"query_points": samples.points, "query_sdf": samples.sdf}
        if samples.normals is not None:
            arrays["normals"] = samples.normals
        np.savez_compressed(path, **arrays)
    elif suffix == ".csv":
        if samples.normals is None:
            values = np.column_stack([samples.points, samples.sdf])
            header = "x,y,z,sdf"
        else:
            values = np.column_stack([samples.points, samples.sdf, samples.normals])
            header = "x,y,z,sdf,nx,ny,nz"
        np.savetxt(path, values, delimiter=",", header=header, comments="")
    else:
        raise ValueError(f"Unsupported SDF sample extension {suffix!r}. Use .npz or .csv.")

    sidecar_metadata = metadata or samples.metadata
    if sidecar_metadata is not None and write_sidecar:
        write_metadata(sidecar_metadata, metadata_sidecar_path(path))
    return path


def sdf_samples_summary(samples: SDFSamples) -> dict[str, Any]:
    samples = make_sdf_samples(samples.points, samples.sdf, normals=samples.normals)
    result: dict[str, Any] = {
        "points_shape": tuple(int(value) for value in samples.points.shape),
        "sdf_shape": tuple(int(value) for value in samples.sdf.shape),
        "sdf_min": float(np.min(samples.sdf)),
        "sdf_max": float(np.max(samples.sdf)),
        "has_normals": samples.normals is not None,
    }
    return result
