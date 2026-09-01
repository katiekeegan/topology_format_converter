from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata


@dataclass
class FieldSamples:
    """Point samples with one scalar value per point."""

    points: np.ndarray
    values: np.ndarray
    field_name: str = "value"
    normals: Optional[np.ndarray] = None
    metadata: Optional[ConversionMetadata] = None


def _validate_points(points: Any) -> np.ndarray:
    points_array = np.asarray(points, dtype=np.float32)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3]; got {points_array.shape}.")
    return points_array


def _validate_values(values: Any, count: int) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float32).reshape(-1)
    if values_array.shape != (count,):
        raise ValueError(
            f"values must have shape ({count},); got {values_array.shape}."
        )
    return values_array


def _validate_normals(normals: Any, points_shape: tuple[int, int]) -> np.ndarray:
    normals_array = np.asarray(normals, dtype=np.float32)
    if normals_array.shape != points_shape:
        raise ValueError(
            f"normals must have shape {points_shape}; got {normals_array.shape}."
        )
    return normals_array


def make_field_samples(
    points: Any,
    values: Any,
    *,
    field_name: str = "value",
    normals: Optional[Any] = None,
    metadata: Optional[ConversionMetadata] = None,
) -> FieldSamples:
    points_array = _validate_points(points)
    values_array = _validate_values(values, len(points_array))
    normals_array = (
        None if normals is None else _validate_normals(normals, points_array.shape)
    )
    return FieldSamples(
        points=points_array,
        values=values_array,
        field_name=str(field_name or "value"),
        normals=normals_array,
        metadata=metadata,
    )


def _scalar_string(value: Any, default: str) -> str:
    try:
        array = np.asarray(value)
        if array.shape == ():
            return str(array.item())
    except Exception:
        pass
    return default


def load_field_samples(
    path: str | Path,
    *,
    points_key: str = "points",
    values_key: str = "values",
    normals_key: Optional[str] = "normals",
    field_name: Optional[str] = None,
) -> FieldSamples:
    """Load scalar point samples from .npz or .csv."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            point_key = points_key
            if point_key not in data:
                for candidate in (
                    "points",
                    "query_points",
                    "coords",
                    "xyz",
                    "surface_points",
                ):
                    if candidate in data:
                        point_key = candidate
                        break
            value_key = values_key
            if value_key not in data:
                for candidate in (
                    "values",
                    "value",
                    "sdf",
                    "query_sdf",
                    "signed_distance",
                    "occupancy",
                    "density",
                    "stress",
                ):
                    if candidate in data:
                        value_key = candidate
                        break
            if point_key not in data:
                raise KeyError(
                    f"Could not find point samples in {path}. Available keys: {data.files}"
                )
            if value_key not in data:
                raise KeyError(
                    f"Could not find scalar field values in {path}. Available keys: {data.files}"
                )
            normals = data[normals_key] if normals_key and normals_key in data else None
            stored_field_name = (
                _scalar_string(data["field_name"], value_key)
                if "field_name" in data
                else value_key
            )
            return make_field_samples(
                data[point_key],
                data[value_key],
                field_name=field_name or stored_field_name,
                normals=normals,
            )

    if suffix == ".csv":
        try:
            named = np.genfromtxt(path, delimiter=",", names=True, dtype=np.float32)
            if named.dtype.names and {"x", "y", "z"}.issubset(named.dtype.names):
                points = np.column_stack([named["x"], named["y"], named["z"]])
                value_name = values_key if values_key in named.dtype.names else None
                if value_name is None:
                    for candidate in (
                        "value",
                        "values",
                        "sdf",
                        "signed_distance",
                        "occupancy",
                        "density",
                        "stress",
                    ):
                        if candidate in named.dtype.names:
                            value_name = candidate
                            break
                if value_name is None:
                    raise ValueError("CSV field samples need a scalar value column.")
                normals = None
                if {"nx", "ny", "nz"}.issubset(named.dtype.names):
                    normals = np.column_stack([named["nx"], named["ny"], named["nz"]])
                return make_field_samples(
                    points,
                    named[value_name],
                    field_name=field_name or value_name,
                    normals=normals,
                )
        except ValueError:
            pass
        values = np.loadtxt(path, delimiter=",", dtype=np.float32)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.shape[1] < 4:
            raise ValueError(
                f"CSV field samples must have at least 4 columns; got {values.shape[1]}."
            )
        normals = values[:, 4:7] if values.shape[1] >= 7 else None
        return make_field_samples(
            values[:, :3],
            values[:, 3],
            field_name=field_name or "value",
            normals=normals,
        )

    raise ValueError(
        f"Unsupported field sample extension {suffix!r}. Use .npz or .csv."
    )


def save_field_samples(
    samples: FieldSamples,
    path: str | Path,
    *,
    metadata: Optional[ConversionMetadata | dict[str, Any]] = None,
    write_sidecar: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = make_field_samples(
        samples.points,
        samples.values,
        field_name=samples.field_name,
        normals=samples.normals,
    )
    suffix = path.suffix.lower()
    if suffix == ".npz":
        arrays: dict[str, Any] = {
            "points": samples.points,
            "values": samples.values,
            "field_name": np.asarray(samples.field_name),
        }
        if samples.normals is not None:
            arrays["normals"] = samples.normals
        np.savez_compressed(path, **arrays)
    elif suffix == ".csv":
        value_name = samples.field_name if samples.field_name else "value"
        if samples.normals is None:
            values = np.column_stack([samples.points, samples.values])
            header = f"x,y,z,{value_name}"
        else:
            values = np.column_stack([samples.points, samples.values, samples.normals])
            header = f"x,y,z,{value_name},nx,ny,nz"
        np.savetxt(path, values, delimiter=",", header=header, comments="")
    else:
        raise ValueError(
            f"Unsupported field sample extension {suffix!r}. Use .npz or .csv."
        )

    sidecar_metadata = metadata or samples.metadata
    if sidecar_metadata is not None and write_sidecar:
        write_metadata(sidecar_metadata, metadata_sidecar_path(path))
    return path


def field_samples_summary(samples: FieldSamples) -> dict[str, Any]:
    samples = make_field_samples(
        samples.points,
        samples.values,
        field_name=samples.field_name,
        normals=samples.normals,
    )
    return {
        "points_shape": tuple(int(value) for value in samples.points.shape),
        "values_shape": tuple(int(value) for value in samples.values.shape),
        "field_name": samples.field_name,
        "value_min": float(np.min(samples.values)),
        "value_max": float(np.max(samples.values)),
        "has_normals": samples.normals is not None,
    }
