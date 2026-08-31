from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Optional


@dataclass
class ConversionMetadata:
    """Metadata sidecar for lossy topology/mesh/cache conversions."""

    source_format: str
    target_format: str
    source_path: Optional[str] = None
    dataset: Optional[str] = None
    split: Optional[str] = None
    index: Optional[int] = None
    threshold: Optional[float] = None
    voxel_shape: Optional[tuple[int, int, int]] = None
    spacing: Optional[tuple[float, float, float]] = None
    coordinate_mode: Optional[str] = None
    sdf_sign: str = "negative_inside"
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, drop_none: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if drop_none:
            data = {key: value for key, value in data.items() if value is not None}
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversionMetadata":
        normalized = dict(data)
        for key in ("voxel_shape", "spacing"):
            if key in normalized and normalized[key] is not None:
                normalized[key] = tuple(normalized[key])
        return cls(**normalized)


def metadata_sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_suffix(path.suffix + ".json")


def write_metadata(metadata: ConversionMetadata | dict[str, Any], path: str | Path) -> Path:
    """Write metadata JSON and return the written path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(metadata, ConversionMetadata):
        data = metadata.to_dict()
    else:
        data = metadata
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_metadata(path: str | Path) -> ConversionMetadata:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ConversionMetadata.from_dict(data)
