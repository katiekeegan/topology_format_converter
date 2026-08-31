from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .cache import density_to_training_cache, save_training_cache
from .mesh import convert_mesh, density_to_mesh, export_mesh, load_mesh, mesh_summary
from .metadata import ConversionMetadata
from .selto import load_selto_sample, selto_sample_to_cache, selto_sample_to_mesh
from .volume import load_volume, signed_distance_from_density


def _spacing(values: Optional[Iterable[str]]) -> Optional[tuple[float, float, float]]:
    if values is None:
        return None
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("--spacing requires exactly 3 values")
    return parsed


def _add_common_volume_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--coordinate-mode", choices=("voxel", "unit-box", "training"), default="training")
    parser.add_argument("--spacing", nargs=3, metavar=("DX", "DY", "DZ"))


def _add_cache_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-surface-points", type=int, default=10000)
    parser.add_argument("--num-query-points", type=int, default=10000)
    parser.add_argument("--surface-sample-ratio", type=float, default=0.7)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)


def cmd_selto_to_mesh(args: argparse.Namespace) -> int:
    sample = load_selto_sample(
        root=args.root,
        dataset=args.dataset,
        split=args.split,
        index=args.index,
        verbose=not args.quiet,
    )
    mesh = selto_sample_to_mesh(
        sample,
        threshold=args.threshold,
        coordinate_mode=args.coordinate_mode,
        repair=not args.no_repair,
    )
    metadata = ConversionMetadata(
        source_format="SELTO/DL4TO",
        target_format=Path(args.out).suffix.lower().lstrip(".") or "mesh",
        dataset=args.dataset,
        split=args.split,
        index=args.index,
        threshold=args.threshold,
        voxel_shape=tuple(int(value) for value in sample.density.shape),
        spacing=sample.voxel_size if args.coordinate_mode == "voxel" else None,
        coordinate_mode=args.coordinate_mode,
        extra={"mesh": mesh_summary(mesh)},
    )
    export_mesh(mesh, args.out, metadata=metadata, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_volume_to_mesh(args: argparse.Namespace) -> int:
    density = load_volume(args.input, key=args.input_key)
    spacing = _spacing(args.spacing)
    mesh = density_to_mesh(
        density,
        threshold=args.threshold,
        coordinate_mode=args.coordinate_mode,
        spacing=spacing,
        repair=not args.no_repair,
    )
    metadata = ConversionMetadata(
        source_format=Path(args.input).suffix.lower().lstrip("."),
        target_format=Path(args.out).suffix.lower().lstrip(".") or "mesh",
        source_path=str(args.input),
        threshold=args.threshold,
        voxel_shape=tuple(int(value) for value in density.shape),
        spacing=spacing if args.coordinate_mode == "voxel" else None,
        coordinate_mode=args.coordinate_mode,
        extra={"mesh": mesh_summary(mesh)},
    )
    export_mesh(mesh, args.out, metadata=metadata, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_selto_to_cache(args: argparse.Namespace) -> int:
    sample = load_selto_sample(
        root=args.root,
        dataset=args.dataset,
        split=args.split,
        index=args.index,
        verbose=not args.quiet,
    )
    cache = selto_sample_to_cache(
        sample,
        threshold=args.threshold,
        coordinate_mode=args.coordinate_mode,
        num_surface_points=args.num_surface_points,
        num_query_points=args.num_query_points,
        surface_sample_ratio=args.surface_sample_ratio,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    save_training_cache(cache, args.out, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_volume_to_cache(args: argparse.Namespace) -> int:
    density = load_volume(args.input, key=args.input_key)
    cache = density_to_training_cache(
        density,
        threshold=args.threshold,
        coordinate_mode=args.coordinate_mode,
        spacing=_spacing(args.spacing),
        num_surface_points=args.num_surface_points,
        num_query_points=args.num_query_points,
        surface_sample_ratio=args.surface_sample_ratio,
        noise_std=args.noise_std,
        seed=args.seed,
        source_format=Path(args.input).suffix.lower().lstrip("."),
        source_path=str(args.input),
    )
    save_training_cache(cache, args.out, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_mesh_convert(args: argparse.Namespace) -> int:
    metadata = ConversionMetadata(
        source_format=Path(args.input).suffix.lower().lstrip("."),
        target_format=Path(args.out).suffix.lower().lstrip(".") or "mesh",
        source_path=str(args.input),
    )
    convert_mesh(args.input, args.out, metadata=metadata, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.input)
    suffix = path.suffix.lower()
    if suffix in {".obj", ".stl", ".ply", ".glb", ".gltf", ".off"}:
        mesh = load_mesh(path)
        print(mesh_summary(mesh))
        return 0

    density = load_volume(path, key=args.input_key)
    sdf = signed_distance_from_density(density, threshold=args.threshold)
    print(
        {
            "shape": tuple(int(value) for value in density.shape),
            "density_min": float(np.min(density)),
            "density_max": float(np.max(density)),
            "solid_voxels": int(np.count_nonzero(density >= args.threshold)),
            "sdf_min": float(np.min(sdf)),
            "sdf_max": float(np.max(sdf)),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gto-convert", description="Convert topology optimization volumes, SELTO samples, SDF caches, and meshes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    selto_mesh = subparsers.add_parser("selto-to-mesh", help="Export one SELTO sample as OBJ/STL/PLY/etc.")
    selto_mesh.add_argument("--root", default=".")
    selto_mesh.add_argument("--dataset", required=True)
    selto_mesh.add_argument("--split", choices=("train", "test"), default="train")
    selto_mesh.add_argument("--index", type=int, default=0)
    selto_mesh.add_argument("--out", required=True)
    selto_mesh.add_argument("--no-repair", action="store_true")
    selto_mesh.add_argument("--no-metadata", action="store_true")
    selto_mesh.add_argument("--quiet", action="store_true")
    _add_common_volume_args(selto_mesh)
    selto_mesh.set_defaults(func=cmd_selto_to_mesh)

    volume_mesh = subparsers.add_parser("volume-to-mesh", help="Export a .npy/.npz/.pt volume as OBJ/STL/PLY/etc.")
    volume_mesh.add_argument("input")
    volume_mesh.add_argument("--out", required=True)
    volume_mesh.add_argument("--input-key", default=None)
    volume_mesh.add_argument("--no-repair", action="store_true")
    volume_mesh.add_argument("--no-metadata", action="store_true")
    _add_common_volume_args(volume_mesh)
    volume_mesh.set_defaults(func=cmd_volume_to_mesh)

    selto_cache = subparsers.add_parser("selto-to-cache", help="Build an SDF training cache from one SELTO sample.")
    selto_cache.add_argument("--root", default=".")
    selto_cache.add_argument("--dataset", required=True)
    selto_cache.add_argument("--split", choices=("train", "test"), default="train")
    selto_cache.add_argument("--index", type=int, default=0)
    selto_cache.add_argument("--out", required=True)
    selto_cache.add_argument("--no-metadata", action="store_true")
    selto_cache.add_argument("--quiet", action="store_true")
    _add_common_volume_args(selto_cache)
    _add_cache_args(selto_cache)
    selto_cache.set_defaults(func=cmd_selto_to_cache)

    volume_cache = subparsers.add_parser("volume-to-cache", help="Build an SDF training cache from a .npy/.npz/.pt volume.")
    volume_cache.add_argument("input")
    volume_cache.add_argument("--out", required=True)
    volume_cache.add_argument("--input-key", default=None)
    volume_cache.add_argument("--no-metadata", action="store_true")
    _add_common_volume_args(volume_cache)
    _add_cache_args(volume_cache)
    volume_cache.set_defaults(func=cmd_volume_to_cache)

    mesh_convert = subparsers.add_parser("mesh-convert", help="Convert between mesh formats supported by trimesh.")
    mesh_convert.add_argument("input")
    mesh_convert.add_argument("--out", required=True)
    mesh_convert.add_argument("--no-metadata", action="store_true")
    mesh_convert.set_defaults(func=cmd_mesh_convert)

    inspect = subparsers.add_parser("inspect", help="Print basic mesh or volume statistics.")
    inspect.add_argument("input")
    inspect.add_argument("--input-key", default=None)
    inspect.add_argument("--threshold", type=float, default=0.5)
    inspect.set_defaults(func=cmd_inspect)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
