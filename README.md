# Topology Format Converter

`topology_format_converter` is a small Python package for converting topology
optimization shape data between signed-distance function (SDF), voxel,
mesh, surface point-cloud, and training-cache representations.

The core package is not tied to any one dataset or project. Plain `.npy`,
`.npz`, `.pt`, OBJ, STL, PLY, CSV, and VTK-style workflows are first-class.
SELTO/DL4TO are particular supported instances: the package includes dedicated
adapter commands for those samples, but the conversion utilities are meant for
general topology optimization data.

The package is intentionally explicit about lossy conversions. Dataset-backed
topology samples can contain more than a surface: density values, loads,
boundary conditions, design-space masks, voxel spacing, split names, and sample
indices may all matter. Mesh exports such as OBJ, STL, and PLY cannot preserve
all of that information, so this package writes JSON metadata sidecars by
default.

## Install

From this repository:

```bash
pip install -e .
```

On HPC systems where dependencies come from environment modules:

```bash
pip install -e . --no-deps
```

The generic volume, mesh, point-cloud, SDF, cache, and VTK commands do not
require DL4TO. Only the `selto-*` commands need it. If you are using a vendored
DL4TO package from another project, include it on `PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/DL4TO-parent:.
```

## Data Modalities

The package distinguishes representation from file extension. This matters
because `.npz` can contain a density grid, occupancy grid, SDF grid, SDF point
samples, a surface point cloud, or a training cache. When the file is
ambiguous, pass `--source-modality`, `--target-modality`, and key names such as
`--input-key`, `--points-key`, or `--sdf-key`.

Supported modalities:

```text
volume          3-D scalar grid, usually density
occupancy       3-D binary voxel grid
sdf-grid        dense 3-D signed-distance or distance grid
sdf-samples     point samples with SDF values, e.g. query_points/query_sdf
mesh            OBJ/STL/PLY/OFF/GLB/GLTF mesh
pointcloud      surface points with optional normals
training-cache  .npz bundle with density, SDF grid, surface points, query points
```

## Supported Conversions

```text
SELTO/DL4TO sample -> OBJ/STL/PLY mesh
SELTO/DL4TO sample -> .npy/.npz density volume
SELTO/DL4TO sample -> SDF training cache (.npz)
SELTO/DL4TO sample range -> batch mesh/cache/volume/SDF export
volume/occupancy -> OBJ/STL/PLY/OFF/GLB/GLTF mesh
volume/occupancy -> signed-distance grid (.npy/.npz)
volume/occupancy -> SDF training cache (.npz)
volume/occupancy -> sampled surface point cloud (.npz/.csv/.ply)
volume/occupancy -> legacy .vtk scalar grid for ParaView
sdf-grid -> zero-level mesh
sdf-grid -> sampled zero-level point cloud
sdf-samples -> SDF samples (.npz/.csv)
sdf-samples -> point cloud positions (.npz/.csv/.ply)
mesh -> mesh format supported by trimesh
mesh -> sampled surface point cloud (.npz/.csv/.ply)
mesh -> surface occupancy grid (.npy/.npz)
mesh -> unsigned nearest-surface distance grid (.npy/.npz)
pointcloud -> point cloud format conversion (.npz/.csv/.ply)
pointcloud -> surface occupancy grid (.npy/.npz)
pointcloud -> unsigned nearest-surface distance grid (.npy/.npz)
training-cache -> mesh, point cloud, SDF samples, SDF grid, or volume
training-cache -> cache summary
```

Converting a mesh or surface point cloud to an SDF grid produces an unsigned
nearest-surface distance grid unless a signed volumetric representation is
already available. A point cloud alone usually does not contain reliable
inside/outside information.

## Command Line

After installation, use:

```bash
topology-convert --help
```

Without installation:

```bash
python -m topology_format_converter.cli --help
```

Export one SELTO sample as a mesh:

```bash
topology-convert selto-to-mesh \
  --root . \
  --dataset sphere_complex \
  --split train \
  --index 0 \
  --coordinate-mode training \
  --out sphere_complex_0.obj
```

Build an SDF training cache from one SELTO sample:

```bash
topology-convert selto-to-cache \
  --root . \
  --dataset disc_simple \
  --index 0 \
  --num-surface-points 10000 \
  --num-query-points 10000 \
  --out disc_simple_0_cache.npz
```

Export one SELTO sample as a density volume:

```bash
topology-convert selto-to-volume \
  --root . \
  --dataset disc_simple \
  --index 0 \
  --out disc_simple_0_density.npz
```

Include SELTO/DL4TO problem arrays in the `.npz` when needed:

```bash
topology-convert selto-to-volume \
  --root . \
  --dataset disc_simple \
  --index 0 \
  --include-problem-fields \
  --out disc_simple_0_fields.npz
```

Convert a volume file to a mesh:

```bash
topology-convert volume-to-mesh density.npy \
  --threshold 0.5 \
  --coordinate-mode unit-box \
  --out density.stl
```

Convert a volume to a signed-distance grid:

```bash
topology-convert volume-to-sdf density.npy \
  --threshold 0.5 \
  --normalize-sdf \
  --out density_sdf.npz
```

Export a volume to legacy VTK for ParaView:

```bash
topology-convert volume-to-vtk density.npy \
  --scalar-name density \
  --out density.vtk
```

Rebuild a mesh from a saved SDF training cache:

```bash
topology-convert cache-to-mesh sphere_complex_0_cache.npz \
  --out sphere_complex_0_from_cache.obj
```

Sample mesh surface points and normals:

```bash
topology-convert mesh-to-pointcloud sphere_complex_0.obj \
  --num-points 10000 \
  --out sphere_complex_0_points.npz
```

Use the general conversion dispatcher when the source and target modalities are
known:

```bash
topology-convert convert-file density.npz \
  --source-modality volume \
  --target-modality mesh \
  --input-key density \
  --threshold 0.5 \
  --coordinate-mode unit-box \
  --out density.obj

topology-convert convert-file density.obj \
  --source-modality mesh \
  --target-modality pointcloud \
  --num-points 10000 \
  --out density_points.csv

topology-convert convert-file density_points.csv \
  --source-modality pointcloud \
  --target-modality occupancy \
  --resolution 64 \
  --mark-radius 1 \
  --out density_surface_occupancy.npz

topology-convert convert-file cache.npz \
  --source-modality training-cache \
  --target-modality sdf-samples \
  --out query_sdf.csv
```

Batch-convert a range of SELTO/DL4TO samples:

```bash
topology-convert selto-batch \
  --root . \
  --dataset sphere_complex \
  --split train \
  --start 0 \
  --count 25 \
  --kind mesh \
  --out-dir exports/sphere_complex_meshes
```

Inspect a mesh, volume, or cache:

```bash
topology-convert inspect density.npy
topology-convert inspect density.obj
topology-convert cache-inspect sphere_complex_0_cache.npz
```

## Python API

```python
from topology_format_converter import (
    cache_to_mesh,
    convert_file,
    density_to_training_cache,
    export_mesh,
    load_pointcloud,
    load_selto_sample,
    make_sdf_samples,
    mesh_to_pointcloud,
    pointcloud_to_distance_grid,
    pointcloud_to_occupancy,
    save_sdf_samples,
    save_training_cache,
    save_volume,
    signed_distance_from_density,
)
from topology_format_converter.selto import selto_sample_to_mesh

sample = load_selto_sample(root=".", dataset="sphere_complex", split="train", index=0)
mesh = selto_sample_to_mesh(sample, threshold=0.5, coordinate_mode="training")
export_mesh(mesh, "sphere_complex_0.obj", metadata=sample.metadata)

save_volume("sphere_complex_0_density.npz", sample.density)
sdf_grid = signed_distance_from_density(sample.density, threshold=0.5)

cache = density_to_training_cache(sample.density, threshold=0.5)
save_training_cache(cache, "sphere_complex_0_cache.npz")

mesh_from_cache = cache_to_mesh(cache)
points, normals = mesh_to_pointcloud(mesh, num_points=10000)

occupancy = pointcloud_to_occupancy(points, shape=(64, 64, 64), coordinate_mode="training")
distance_grid = pointcloud_to_distance_grid(points, shape=(64, 64, 64), coordinate_mode="training")

sdf_samples = make_sdf_samples(cache.query_points, cache.query_sdf)
save_sdf_samples(sdf_samples, "query_sdf.csv")

convert_file(
    "sphere_complex_0_density.npz",
    "sphere_complex_0.obj",
    source_modality="volume",
    target_modality="mesh",
    input_key="density",
)
```

## Coordinate Modes

- `voxel`: preserve voxel-index units, optionally with `--spacing DX DY DZ`.
- `unit-box`: map each axis independently into `[-1, 1]`.
- `training`: use the normalization convention commonly used for SDF model
  training: the longest voxel axis maps to length 2.

## Metadata

Mesh, volume, SDF, cache, point-cloud, and VTK exports write a sidecar unless
`--no-metadata` is passed.
Examples:

```text
sample.obj
sample.obj.json
sample_cache.npz
sample_cache.npz.json
density.vtk
density.vtk.json
```

The sidecar records source format, target format, dataset, split, sample index,
density threshold, voxel shape, spacing, coordinate mode, and SDF sign
convention.

## Development

Run tests:

```bash
python -m pytest tests -q
```
