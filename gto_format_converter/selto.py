from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np

from .cache import TrainingCache, density_to_training_cache
from .mesh import density_to_mesh
from .metadata import ConversionMetadata
from .volume import to_numpy


@dataclass
class TopologySample:
    """A SELTO/DL4TO topology optimization sample with canonical arrays."""

    density: np.ndarray
    force: Optional[np.ndarray]
    design_space: Optional[np.ndarray]
    dirichlet: Optional[np.ndarray]
    voxel_size: Optional[tuple[float, float, float]]
    metadata: ConversionMetadata
    problem: Optional[Any] = None
    solution: Optional[Any] = None


def _import_selto_dataset():
    try:
        from dl4to.datasets import SELTODataset

        return SELTODataset
    except ImportError as exc:
        raise ImportError(
            "dl4to is required for SELTO loading. Install DL4TO or run with "
            "PYTHONPATH pointing at the vendored dl4to package."
        ) from exc


def _optional_array(obj: Any, attr: str) -> Optional[np.ndarray]:
    if not hasattr(obj, attr):
        return None
    return np.asarray(to_numpy(getattr(obj, attr))).copy()


def _voxel_size(problem: Any) -> Optional[tuple[float, float, float]]:
    h = getattr(problem, "h", None)
    if h is None:
        return None
    values = np.asarray(to_numpy(h), dtype=np.float32).reshape(-1)
    if values.size == 1:
        return (float(values[0]), float(values[0]), float(values[0]))
    if values.size >= 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    return None


def load_selto_dataset(
    *,
    root: str = ".",
    dataset: str,
    split: str = "train",
    size: int = -1,
    download: bool = True,
    verbose: bool = True,
) -> Any:
    """Load a DL4TO SELTO dataset object."""

    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'.")
    SELTODataset = _import_selto_dataset()
    return SELTODataset(
        root=root,
        name=dataset,
        train=(split == "train"),
        size=size,
        download=download,
        verbose=verbose,
    )


def load_selto_sample(
    *,
    root: str = ".",
    dataset: str,
    split: str = "train",
    index: int = 0,
    download: bool = True,
    verbose: bool = True,
) -> TopologySample:
    """Load one SELTO sample as numpy arrays plus optional DL4TO objects."""

    if index < 0:
        raise ValueError("index must be non-negative.")
    selto = load_selto_dataset(
        root=root,
        dataset=dataset,
        split=split,
        size=index + 1,
        download=download,
        verbose=verbose,
    )
    if index >= len(selto):
        raise IndexError(f"SELTO sample index {index} is unavailable; loaded {len(selto)} samples.")

    problem, solution = selto[index]
    density = np.asarray(to_numpy(getattr(solution, "\u03b8"))).squeeze().astype(np.float32, copy=False)
    if density.ndim != 3:
        raise ValueError(f"SELTO density must be 3-D after squeeze; got {density.shape}.")

    metadata = ConversionMetadata(
        source_format="SELTO/DL4TO",
        target_format="sample",
        dataset=dataset,
        split=split,
        index=int(index),
        voxel_shape=tuple(int(value) for value in density.shape),
        spacing=_voxel_size(problem),
        coordinate_mode="voxel",
        extra={"problem_name": getattr(problem, "name", None)},
    )

    return TopologySample(
        density=density,
        force=_optional_array(problem, "F"),
        design_space=_optional_array(problem, "\u03a9_design"),
        dirichlet=_optional_array(problem, "\u03a9_dirichlet"),
        voxel_size=_voxel_size(problem),
        metadata=metadata,
        problem=problem,
        solution=solution,
    )


def iter_selto_samples(
    *,
    root: str = ".",
    dataset: str,
    split: str = "train",
    max_samples: int = -1,
    download: bool = True,
    verbose: bool = True,
) -> Iterator[TopologySample]:
    size = max_samples if max_samples and max_samples > 0 else -1
    selto = load_selto_dataset(
        root=root,
        dataset=dataset,
        split=split,
        size=size,
        download=download,
        verbose=verbose,
    )
    for index in range(len(selto)):
        problem, solution = selto[index]
        density = np.asarray(to_numpy(getattr(solution, "\u03b8"))).squeeze().astype(np.float32, copy=False)
        metadata = ConversionMetadata(
            source_format="SELTO/DL4TO",
            target_format="sample",
            dataset=dataset,
            split=split,
            index=int(index),
            voxel_shape=tuple(int(value) for value in density.shape),
            spacing=_voxel_size(problem),
            coordinate_mode="voxel",
            extra={"problem_name": getattr(problem, "name", None)},
        )
        yield TopologySample(
            density=density,
            force=_optional_array(problem, "F"),
            design_space=_optional_array(problem, "\u03a9_design"),
            dirichlet=_optional_array(problem, "\u03a9_dirichlet"),
            voxel_size=_voxel_size(problem),
            metadata=metadata,
            problem=problem,
            solution=solution,
        )


def selto_sample_to_mesh(
    sample: TopologySample,
    *,
    threshold: float = 0.5,
    coordinate_mode: str = "training",
    repair: bool = True,
):
    return density_to_mesh(
        sample.density,
        threshold=threshold,
        coordinate_mode=coordinate_mode,
        spacing=sample.voxel_size if coordinate_mode == "voxel" else None,
        repair=repair,
    )


def selto_sample_to_cache(
    sample: TopologySample,
    *,
    threshold: float = 0.5,
    coordinate_mode: str = "training",
    num_surface_points: int = 10000,
    num_query_points: int = 10000,
    surface_sample_ratio: float = 0.7,
    noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> TrainingCache:
    cache = density_to_training_cache(
        sample.density,
        threshold=threshold,
        coordinate_mode=coordinate_mode,
        spacing=sample.voxel_size if coordinate_mode == "voxel" else None,
        num_surface_points=num_surface_points,
        num_query_points=num_query_points,
        surface_sample_ratio=surface_sample_ratio,
        noise_std=noise_std,
        seed=seed,
        source_format="SELTO/DL4TO",
        metadata_extra=sample.metadata.to_dict(),
    )
    cache.metadata.dataset = sample.metadata.dataset
    cache.metadata.split = sample.metadata.split
    cache.metadata.index = sample.metadata.index
    cache.metadata.source_path = sample.metadata.source_path
    return cache
