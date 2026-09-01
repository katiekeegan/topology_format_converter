"""Reusable conversion tools for topology optimization shape data.

The package treats voxel density arrays as canonical volume data, then derives
meshes, signed-distance grids, point clouds, and training caches. SELTO/DL4TO
support is provided through dedicated dataset adapters.
"""

from .cache import (
    TrainingCache,
    cache_summary,
    cache_to_mesh,
    density_to_training_cache,
    load_training_cache,
    save_training_cache,
)
from .convert import Modality, convert_file, infer_modality
from .mesh import convert_mesh, density_to_mesh, export_mesh, load_mesh, repair_mesh
from .metadata import ConversionMetadata, read_metadata, write_metadata
from .pointcloud import (
    export_pointcloud,
    load_mesh_as_pointcloud,
    load_pointcloud,
    mesh_to_pointcloud,
    pointcloud_summary,
    pointcloud_to_distance_grid,
    pointcloud_to_occupancy,
)
from .selto import TopologySample, load_selto_dataset, load_selto_sample
from .sdf import SDFSamples, load_sdf_samples, make_sdf_samples, save_sdf_samples, sdf_samples_summary
from .volume import (
    load_volume,
    save_signed_distance,
    save_volume,
    save_volume_arrays_npz,
    save_volume_npz,
    save_volume_vtk,
    signed_distance_from_density,
    threshold_density,
)

__all__ = [
    "ConversionMetadata",
    "Modality",
    "SDFSamples",
    "TopologySample",
    "TrainingCache",
    "cache_summary",
    "cache_to_mesh",
    "convert_mesh",
    "convert_file",
    "density_to_mesh",
    "density_to_training_cache",
    "export_mesh",
    "export_pointcloud",
    "load_mesh_as_pointcloud",
    "load_mesh",
    "load_pointcloud",
    "load_sdf_samples",
    "load_selto_dataset",
    "load_selto_sample",
    "load_training_cache",
    "load_volume",
    "infer_modality",
    "make_sdf_samples",
    "mesh_to_pointcloud",
    "pointcloud_summary",
    "pointcloud_to_distance_grid",
    "pointcloud_to_occupancy",
    "read_metadata",
    "repair_mesh",
    "save_sdf_samples",
    "save_signed_distance",
    "save_training_cache",
    "save_volume",
    "save_volume_arrays_npz",
    "save_volume_npz",
    "save_volume_vtk",
    "sdf_samples_summary",
    "signed_distance_from_density",
    "threshold_density",
    "write_metadata",
]
