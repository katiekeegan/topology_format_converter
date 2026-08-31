# Topology Format Converter

`topology_format_converter` is a small Python package for converting topology
optimization volume data, signed-distance grids, SDF training caches, point
clouds, and meshes.

The core package is not tied to any one dataset or project. Plain `.npy`,
`.npz`, `.pt`, OBJ, STL, PLY, and VTK-style workflows are first-class.
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

## Supported Conversions

```text
SELTO/DL4TO sample -> OBJ/STL/PLY mesh
SELTO/DL4TO sample -> .npy/.npz density volume
SELTO/DL4TO sample -> SDF training cache (.npz)
SELTO/DL4TO sample range -> batch mesh/cache/volume/SDF export
.npy/.npz/.pt volume -> OBJ/STL/PLY mesh
.npy/.npz/.pt volume -> signed-distance grid (.npy/.npz)
.npy/.npz/.pt volume -> SDF training cache (.npz)
.npy/.npz/.pt volume -> legacy .vtk scalar grid for ParaView
SDF training cache (.npz) -> OBJ/STL/PLY mesh
SDF training cache (.npz) -> cache summary
OBJ/STL/PLY mesh -> sampled point cloud (.npz/.csv/.ply)
mesh -> mesh format supported by trimesh
```

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
    density_to_training_cache,
    export_mesh,
    load_selto_sample,
    mesh_to_pointcloud,
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
