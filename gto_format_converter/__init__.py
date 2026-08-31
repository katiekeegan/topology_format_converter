"""Reusable conversion tools for topology optimization shapes.

The package treats SELTO/DL4TO samples and voxel density arrays as canonical
volume data, then derives meshes, signed-distance grids, and training caches.
"""

from .cache import TrainingCache, density_to_training_cache, load_training_cache, save_training_cache
from .mesh import convert_mesh, density_to_mesh, export_mesh, load_mesh, repair_mesh
from .metadata import ConversionMetadata, read_metadata, write_metadata
from .selto import TopologySample, load_selto_dataset, load_selto_sample
from .volume import load_volume, save_volume_npz, signed_distance_from_density, threshold_density

__all__ = [
    "ConversionMetadata",
    "TopologySample",
    "TrainingCache",
    "convert_mesh",
    "density_to_mesh",
    "density_to_training_cache",
    "export_mesh",
    "load_mesh",
    "load_selto_dataset",
    "load_selto_sample",
    "load_training_cache",
    "load_volume",
    "read_metadata",
    "repair_mesh",
    "save_training_cache",
    "save_volume_npz",
    "signed_distance_from_density",
    "threshold_density",
    "write_metadata",
]
