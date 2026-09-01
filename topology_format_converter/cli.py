from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .cache import cache_summary, cache_to_mesh, density_to_training_cache, load_training_cache, save_training_cache
from .convert import Modality, convert_file
from .mesh import convert_mesh, density_to_mesh, export_mesh, load_mesh, mesh_summary
from .metadata import ConversionMetadata, metadata_sidecar_path, write_metadata
from .pointcloud import export_pointcloud, load_mesh_as_pointcloud, load_pointcloud, pointcloud_summary
from .selto import iter_selto_samples, load_selto_sample, selto_sample_to_cache, selto_sample_to_mesh
from .sdf import load_sdf_samples, sdf_samples_summary
from .volume import (
    load_volume,
    save_signed_distance,
    save_volume,
    save_volume_arrays_npz,
    save_volume_vtk,
    sdf_normalization_scale,
    signed_distance_from_density,
)


def _spacing(values: Optional[Iterable[str]]) -> Optional[tuple[float, float, float]]:
    if values is None:
        return None
    parsed = tuple(float(value) for value in values)
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError("--spacing requires exactly 3 values")
    return parsed


def _print_json(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def _source_format(path: str | Path) -> str:
    return Path(path).suffix.lower().lstrip(".") or "unknown"


def _target_format(path: str | Path) -> str:
    return Path(path).suffix.lower().lstrip(".") or "unknown"


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


def _add_general_conversion_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-modality", choices=MODALITY_CHOICES, default=None)
    parser.add_argument("--target-modality", choices=MODALITY_CHOICES, default=None)
    parser.add_argument("--input-key", default=None)
    parser.add_argument("--points-key", default="points")
    parser.add_argument("--normals-key", default="normals")
    parser.add_argument("--sdf-key", default="query_sdf")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--coordinate-mode", choices=("voxel", "unit-box", "training"), default="unit-box")
    parser.add_argument("--spacing", nargs=3, metavar=("DX", "DY", "DZ"))
    parser.add_argument("--resolution", nargs="+", type=int, default=[64])
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument("--mark-radius", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")


MODALITY_CHOICES = ("volume", "occupancy", "sdf-grid", "sdf-samples", "mesh", "pointcloud", "training-cache")


def _selto_metadata(
    args: argparse.Namespace,
    sample,
    target_format: str,
    *,
    threshold: Optional[float] = None,
    coordinate_mode: Optional[str] = None,
    extra: Optional[dict] = None,
) -> ConversionMetadata:
    return ConversionMetadata(
        source_format="SELTO/DL4TO",
        target_format=target_format,
        dataset=args.dataset,
        split=args.split,
        index=sample.metadata.index,
        threshold=threshold,
        voxel_shape=tuple(int(value) for value in sample.density.shape),
        spacing=sample.voxel_size if coordinate_mode == "voxel" else None,
        coordinate_mode=coordinate_mode,
        extra=extra or {},
    )


def _save_selto_volume(sample, args: argparse.Namespace, out_path: str | Path) -> Path:
    metadata = _selto_metadata(args, sample, _target_format(out_path))
    if args.include_problem_fields:
        if Path(out_path).suffix.lower() != ".npz":
            raise ValueError("--include-problem-fields requires an .npz output path.")
        arrays = {"density": sample.density}
        if sample.force is not None:
            arrays["force"] = sample.force
        if sample.design_space is not None:
            arrays["design_space"] = sample.design_space
        if sample.dirichlet is not None:
            arrays["dirichlet"] = sample.dirichlet
        return save_volume_arrays_npz(
            out_path,
            arrays,
            metadata=metadata,
            write_sidecar=not args.no_metadata,
        )
    return save_volume(
        out_path,
        sample.density,
        key="density",
        metadata=metadata,
        write_sidecar=not args.no_metadata,
    )


def _sdf_from_density(density, args: argparse.Namespace, spacing=None) -> np.ndarray:
    sdf = signed_distance_from_density(density, threshold=args.threshold, spacing=spacing)
    if getattr(args, "normalize_sdf", False):
        scale = sdf_normalization_scale(density.shape, spacing=spacing)
        if scale > 0:
            sdf = sdf / scale
    return sdf


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
    metadata = _selto_metadata(
        args,
        sample,
        _target_format(args.out),
        threshold=args.threshold,
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
        source_format=_source_format(args.input),
        target_format=_target_format(args.out),
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


def cmd_selto_to_volume(args: argparse.Namespace) -> int:
    sample = load_selto_sample(
        root=args.root,
        dataset=args.dataset,
        split=args.split,
        index=args.index,
        verbose=not args.quiet,
    )
    _save_selto_volume(sample, args, args.out)
    print(args.out)
    return 0


def cmd_volume_to_sdf(args: argparse.Namespace) -> int:
    density = load_volume(args.input, key=args.input_key)
    spacing = _spacing(args.spacing)
    metadata = ConversionMetadata(
        source_format=_source_format(args.input),
        target_format=_target_format(args.out),
        source_path=str(args.input),
        threshold=args.threshold,
        voxel_shape=tuple(int(value) for value in density.shape),
        spacing=spacing,
        extra={"array": "sdf", "normalize_sdf": bool(args.normalize_sdf)},
    )
    save_signed_distance(
        args.out,
        density,
        threshold=args.threshold,
        spacing=spacing,
        normalize=args.normalize_sdf,
        metadata=metadata,
        write_sidecar=not args.no_metadata,
    )
    print(args.out)
    return 0


def cmd_volume_to_vtk(args: argparse.Namespace) -> int:
    volume = load_volume(args.input, key=args.input_key)
    spacing = _spacing(args.spacing)
    save_volume_vtk(args.out, volume, scalar_name=args.scalar_name, spacing=spacing)
    if not args.no_metadata:
        metadata = ConversionMetadata(
            source_format=_source_format(args.input),
            target_format=_target_format(args.out),
            source_path=str(args.input),
            voxel_shape=tuple(int(value) for value in volume.shape),
            spacing=spacing,
            extra={"scalar_name": args.scalar_name},
        )
        write_metadata(metadata, metadata_sidecar_path(args.out))
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
        source_format=_source_format(args.input),
        source_path=str(args.input),
    )
    save_training_cache(cache, args.out, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_cache_to_mesh(args: argparse.Namespace) -> int:
    cache = load_training_cache(args.input)
    mesh = cache_to_mesh(
        cache,
        threshold=args.threshold,
        coordinate_mode=args.coordinate_mode,
        spacing=_spacing(args.spacing),
        repair=not args.no_repair,
    )
    metadata = ConversionMetadata(
        source_format="training-cache",
        target_format=_target_format(args.out),
        source_path=str(args.input),
        threshold=args.threshold if args.threshold is not None else cache.metadata.threshold,
        voxel_shape=tuple(int(value) for value in cache.density.shape),
        spacing=_spacing(args.spacing) if args.coordinate_mode == "voxel" else None,
        coordinate_mode=args.coordinate_mode or cache.metadata.coordinate_mode,
        extra={"mesh": mesh_summary(mesh)},
    )
    export_mesh(mesh, args.out, metadata=metadata, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_cache_inspect(args: argparse.Namespace) -> int:
    _print_json(cache_summary(load_training_cache(args.input)))
    return 0


def cmd_mesh_to_pointcloud(args: argparse.Namespace) -> int:
    points, normals = load_mesh_as_pointcloud(args.input, num_points=args.num_points, seed=args.seed)
    metadata = ConversionMetadata(
        source_format=_source_format(args.input),
        target_format=_target_format(args.out),
        source_path=str(args.input),
        extra={"num_points": int(args.num_points), "arrays": ["points", "normals"]},
    )
    export_pointcloud(points, normals, args.out, metadata=metadata, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_mesh_convert(args: argparse.Namespace) -> int:
    metadata = ConversionMetadata(
        source_format=_source_format(args.input),
        target_format=_target_format(args.out),
        source_path=str(args.input),
    )
    convert_mesh(args.input, args.out, metadata=metadata, write_sidecar=not args.no_metadata)
    print(args.out)
    return 0


def cmd_convert_file(args: argparse.Namespace) -> int:
    resolution: int | tuple[int, int, int]
    if len(args.resolution) == 1:
        resolution = int(args.resolution[0])
    elif len(args.resolution) == 3:
        resolution = tuple(int(value) for value in args.resolution)
    else:
        raise ValueError("--resolution expects either one integer or three integers.")
    normals_key = None if args.normals_key.lower() in {"", "none", "null"} else args.normals_key
    convert_file(
        args.input,
        args.out,
        source_modality=args.source_modality,
        target_modality=args.target_modality,
        input_key=args.input_key,
        points_key=args.points_key,
        normals_key=normals_key,
        sdf_key=args.sdf_key,
        threshold=args.threshold,
        coordinate_mode=args.coordinate_mode,
        spacing=_spacing(args.spacing),
        resolution=resolution,
        num_points=args.num_points,
        mark_radius=args.mark_radius,
        seed=args.seed,
        no_repair=args.no_repair,
        no_metadata=args.no_metadata,
    )
    print(args.out)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    path = Path(args.input)
    suffix = path.suffix.lower()
    if suffix in {".obj", ".stl", ".ply", ".glb", ".gltf", ".off"}:
        _print_json(mesh_summary(load_mesh(path)))
        return 0
    if suffix == ".npz":
        try:
            _print_json(cache_summary(load_training_cache(path)))
            return 0
        except KeyError:
            pass
        try:
            _print_json(sdf_samples_summary(load_sdf_samples(path)))
            return 0
        except Exception:
            pass
        try:
            points, normals = load_pointcloud(path)
            _print_json(pointcloud_summary(points, normals))
            return 0
        except Exception:
            pass
    if suffix == ".csv":
        try:
            _print_json(sdf_samples_summary(load_sdf_samples(path)))
            return 0
        except Exception:
            pass
        points, normals = load_pointcloud(path)
        _print_json(pointcloud_summary(points, normals))
        return 0

    density = load_volume(path, key=args.input_key)
    sdf = signed_distance_from_density(density, threshold=args.threshold)
    _print_json(
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


def _batch_output_path(args: argparse.Namespace, index: int) -> Path:
    out_dir = Path(args.out_dir)
    if args.kind == "mesh":
        extension = args.mesh_format.lstrip(".")
    elif args.kind == "cache":
        extension = "npz"
    else:
        extension = args.array_format.lstrip(".")
    return out_dir / f"{args.dataset}_{args.split}_{index:06d}.{extension}"


def cmd_selto_batch(args: argparse.Namespace) -> int:
    if args.start < 0:
        raise ValueError("--start must be non-negative.")
    if args.count <= 0:
        raise ValueError("--count must be positive.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = args.start + args.count
    written = []
    for sample in iter_selto_samples(
        root=args.root,
        dataset=args.dataset,
        split=args.split,
        max_samples=max_samples,
        verbose=not args.quiet,
    ):
        index = sample.metadata.index
        if index is None or index < args.start:
            continue
        out_path = _batch_output_path(args, index)
        if args.kind == "mesh":
            mesh = selto_sample_to_mesh(
                sample,
                threshold=args.threshold,
                coordinate_mode=args.coordinate_mode,
                repair=not args.no_repair,
            )
            metadata = _selto_metadata(
                args,
                sample,
                _target_format(out_path),
                threshold=args.threshold,
                coordinate_mode=args.coordinate_mode,
                extra={"mesh": mesh_summary(mesh)},
            )
            export_mesh(mesh, out_path, metadata=metadata, write_sidecar=not args.no_metadata)
        elif args.kind == "cache":
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
            save_training_cache(cache, out_path, write_sidecar=not args.no_metadata)
        elif args.kind == "volume":
            _save_selto_volume(sample, args, out_path)
        elif args.kind == "sdf":
            spacing = sample.voxel_size if args.use_voxel_spacing else None
            sdf = _sdf_from_density(sample.density, args, spacing=spacing)
            metadata = _selto_metadata(
                args,
                sample,
                _target_format(out_path),
                threshold=args.threshold,
                extra={"array": "sdf", "normalize_sdf": bool(args.normalize_sdf)},
            )
            save_volume(out_path, sdf, key="sdf", metadata=metadata, write_sidecar=not args.no_metadata)
        else:
            raise ValueError(f"Unsupported batch kind {args.kind!r}.")
        written.append(str(out_path))

    _print_json({"written": written, "count": len(written)})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="topology-convert",
        description="Convert topology optimization volumes, SELTO/DL4TO samples, SDF caches, point clouds, and meshes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    selto_mesh = subparsers.add_parser("selto-to-mesh", help="Export one SELTO/DL4TO sample as OBJ/STL/PLY/etc.")
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

    selto_volume = subparsers.add_parser("selto-to-volume", help="Export one SELTO/DL4TO sample density as .npy or .npz.")
    selto_volume.add_argument("--root", default=".")
    selto_volume.add_argument("--dataset", required=True)
    selto_volume.add_argument("--split", choices=("train", "test"), default="train")
    selto_volume.add_argument("--index", type=int, default=0)
    selto_volume.add_argument("--out", required=True)
    selto_volume.add_argument("--include-problem-fields", action="store_true")
    selto_volume.add_argument("--no-metadata", action="store_true")
    selto_volume.add_argument("--quiet", action="store_true")
    selto_volume.set_defaults(func=cmd_selto_to_volume)

    volume_mesh = subparsers.add_parser("volume-to-mesh", help="Export a .npy/.npz/.pt volume as OBJ/STL/PLY/etc.")
    volume_mesh.add_argument("input")
    volume_mesh.add_argument("--out", required=True)
    volume_mesh.add_argument("--input-key", default=None)
    volume_mesh.add_argument("--no-repair", action="store_true")
    volume_mesh.add_argument("--no-metadata", action="store_true")
    _add_common_volume_args(volume_mesh)
    volume_mesh.set_defaults(func=cmd_volume_to_mesh)

    volume_sdf = subparsers.add_parser("volume-to-sdf", help="Export a signed-distance grid from a .npy/.npz/.pt volume.")
    volume_sdf.add_argument("input")
    volume_sdf.add_argument("--out", required=True)
    volume_sdf.add_argument("--input-key", default=None)
    volume_sdf.add_argument("--threshold", type=float, default=0.5)
    volume_sdf.add_argument("--spacing", nargs=3, metavar=("DX", "DY", "DZ"))
    volume_sdf.add_argument("--normalize-sdf", action="store_true")
    volume_sdf.add_argument("--no-metadata", action="store_true")
    volume_sdf.set_defaults(func=cmd_volume_to_sdf)

    volume_vtk = subparsers.add_parser("volume-to-vtk", help="Export a scalar volume as legacy .vtk for ParaView.")
    volume_vtk.add_argument("input")
    volume_vtk.add_argument("--out", required=True)
    volume_vtk.add_argument("--input-key", default=None)
    volume_vtk.add_argument("--spacing", nargs=3, metavar=("DX", "DY", "DZ"))
    volume_vtk.add_argument("--scalar-name", default="density")
    volume_vtk.add_argument("--no-metadata", action="store_true")
    volume_vtk.set_defaults(func=cmd_volume_to_vtk)

    selto_cache = subparsers.add_parser("selto-to-cache", help="Build an SDF training cache from one SELTO/DL4TO sample.")
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

    cache_mesh = subparsers.add_parser("cache-to-mesh", help="Rebuild a mesh from a saved SDF training cache density.")
    cache_mesh.add_argument("input")
    cache_mesh.add_argument("--out", required=True)
    cache_mesh.add_argument("--threshold", type=float, default=None)
    cache_mesh.add_argument("--coordinate-mode", choices=("voxel", "unit-box", "training"), default=None)
    cache_mesh.add_argument("--spacing", nargs=3, metavar=("DX", "DY", "DZ"))
    cache_mesh.add_argument("--no-repair", action="store_true")
    cache_mesh.add_argument("--no-metadata", action="store_true")
    cache_mesh.set_defaults(func=cmd_cache_to_mesh)

    cache_inspect = subparsers.add_parser("cache-inspect", help="Summarize a saved SDF training cache.")
    cache_inspect.add_argument("input")
    cache_inspect.set_defaults(func=cmd_cache_inspect)

    pointcloud = subparsers.add_parser("mesh-to-pointcloud", help="Sample mesh surface points and normals as .npz, .csv, or .ply.")
    pointcloud.add_argument("input")
    pointcloud.add_argument("--out", required=True)
    pointcloud.add_argument("--num-points", type=int, default=10000)
    pointcloud.add_argument("--seed", type=int, default=None)
    pointcloud.add_argument("--no-metadata", action="store_true")
    pointcloud.set_defaults(func=cmd_mesh_to_pointcloud)

    mesh_convert = subparsers.add_parser("mesh-convert", help="Convert between mesh formats supported by trimesh.")
    mesh_convert.add_argument("input")
    mesh_convert.add_argument("--out", required=True)
    mesh_convert.add_argument("--no-metadata", action="store_true")
    mesh_convert.set_defaults(func=cmd_mesh_convert)

    general_convert = subparsers.add_parser(
        "convert-file",
        help="Convert one file between volume, occupancy, SDF grid/sample, mesh, point-cloud, and cache modalities.",
        epilog=(
            "Note: mesh or pointcloud to sdf-grid writes an unsigned nearest-surface "
            "distance grid unless signed volumetric information is available."
        ),
    )
    general_convert.add_argument("input")
    general_convert.add_argument("--out", required=True)
    _add_general_conversion_args(general_convert)
    general_convert.set_defaults(func=cmd_convert_file)

    selto_batch = subparsers.add_parser("selto-batch", help="Batch-convert a range of SELTO/DL4TO samples.")
    selto_batch.add_argument("--root", default=".")
    selto_batch.add_argument("--dataset", required=True)
    selto_batch.add_argument("--split", choices=("train", "test"), default="train")
    selto_batch.add_argument("--start", type=int, default=0)
    selto_batch.add_argument("--count", type=int, required=True)
    selto_batch.add_argument("--kind", choices=("mesh", "cache", "volume", "sdf"), required=True)
    selto_batch.add_argument("--out-dir", required=True)
    selto_batch.add_argument("--mesh-format", default="obj")
    selto_batch.add_argument("--array-format", choices=("npy", "npz"), default="npz")
    selto_batch.add_argument("--include-problem-fields", action="store_true")
    selto_batch.add_argument("--use-voxel-spacing", action="store_true")
    selto_batch.add_argument("--normalize-sdf", action="store_true")
    selto_batch.add_argument("--no-repair", action="store_true")
    selto_batch.add_argument("--no-metadata", action="store_true")
    selto_batch.add_argument("--quiet", action="store_true")
    _add_common_volume_args(selto_batch)
    _add_cache_args(selto_batch)
    selto_batch.set_defaults(func=cmd_selto_batch)

    inspect = subparsers.add_parser("inspect", help="Print basic mesh, cache, or volume statistics.")
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
