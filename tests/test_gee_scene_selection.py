"""Unit tests for GEE scene selection helpers.

These tests use small fakes instead of Earth Engine auth.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from omegaconf import OmegaConf

from berlin_lst_downscaling.data import gee_scenes


@dataclass
class FakeImage:
    props: dict[str, object]

    def get(self, key: str) -> object:
        return self.props.get(key)

    def set(self, mapping: dict[str, object]) -> FakeImage:
        new_props = dict(self.props)
        new_props.update(mapping)
        return FakeImage(new_props)


class FakeNumber:
    def __init__(self, value: int) -> None:
        self.value = value

    def getInfo(self) -> int:
        return self.value


class FakeList:
    def __init__(self, values: Sequence[Any]) -> None:
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, idx: int) -> object:
        return self.values[idx]

    def distinct(self) -> FakeList:
        out: list[object] = []
        for value in self.values:
            if value not in out:
                out.append(value)
        return FakeList(out)

    def sort(self) -> FakeList:
        return FakeList(sorted(self.values))

    def map(self, fn):
        return FakeList([fn(item) for item in self.values])

    def get(self, idx: int) -> object:
        return self[idx]


class FakeString:
    def __init__(self, value: object) -> None:
        self.value = str(value)

    def split(self, delimiter: str) -> FakeList:
        return FakeList(self.value.split(delimiter))


class FakeFilter:
    @staticmethod
    def eq(field: str, value: object) -> tuple[str, str, object]:
        return ("eq", field, value)

    @staticmethod
    def And(*filters: object) -> tuple[str, tuple[object, ...]]:
        return ("and", filters)


class FakeCollection:
    def __init__(self, images: Sequence[FakeImage]) -> None:
        self.images = images

    def merge(self, other: FakeCollection) -> FakeCollection:
        return FakeCollection([*self.images, *other.images])

    def filterDate(self, *_args, **_kwargs) -> FakeCollection:
        return self

    def filterBounds(self, *_args, **_kwargs) -> FakeCollection:
        return self

    def filter(self, filter_obj: object) -> FakeCollection:
        return FakeCollection([img for img in self.images if _matches(img, filter_obj)])

    def sort(self, key: str) -> FakeCollection:
        return FakeCollection(sorted(self.images, key=lambda img: str(img.get(key))))

    def first(self) -> FakeImage:
        return self.images[0]

    def mosaic(self) -> FakeImage:
        return FakeImage(dict(self.images[0].props))

    def aggregate_array(self, key: str) -> list[object]:
        return [img.get(key) for img in self.images]

    def select(self, *_args, **_kwargs) -> FakeCollection:
        return self

    def toList(self, n: int) -> FakeList:
        return FakeList(self.images[:n])

    def size(self) -> FakeNumber:
        return FakeNumber(len(self.images))


class FakeEE:
    def __init__(self, registry: dict[str, FakeCollection]) -> None:
        self.registry = registry
        self.Filter = FakeFilter

    def ImageCollection(self, value):
        if isinstance(value, str):
            return self.registry[value]
        if isinstance(value, FakeList):
            return FakeCollection(list(value.values))
        return value

    def List(self, value):
        return FakeList(value)

    def String(self, value):
        return FakeString(value)

    def Image(self, value):
        return value


def _matches(img: FakeImage, filter_obj: object) -> bool:
    if not isinstance(filter_obj, tuple):
        return True

    kind = filter_obj[0]
    if kind == "eq":
        _, field, value = filter_obj
        return img.get(field) == value
    if kind == "and":
        _, filters = filter_obj
        return all(_matches(img, sub) for sub in filters)
    return True


def test_list_landsat_scenes_uses_explicit_path_row_allowlist(monkeypatch) -> None:
    registry = {
        "LANDSAT/LC08/C02/T1_L2": FakeCollection(
            [
                FakeImage({"system:index": "keep-1", "WRS_PATH": 193, "WRS_ROW": 23}),
                FakeImage({"system:index": "drop-1", "WRS_PATH": 192, "WRS_ROW": 23}),
            ]
        ),
        "LANDSAT/LC09/C02/T1_L2": FakeCollection(
            [
                FakeImage({"system:index": "keep-2", "WRS_PATH": 193, "WRS_ROW": 23}),
                FakeImage({"system:index": "drop-2", "WRS_PATH": 193, "WRS_ROW": 24}),
            ]
        ),
    }
    monkeypatch.setattr(gee_scenes, "ee", FakeEE(registry))
    cfg = OmegaConf.create(
        {
            "ard": {"time": {"start_year": 2023, "end_year": 2023, "months": [5]}},
            "landsat": {
                "collections": list(registry),
                "scene_filter": {"wrs_paths": [193], "wrs_rows": [23]},
            },
        }
    )

    collection = cast(FakeCollection, gee_scenes.list_landsat_scenes(cfg, year=2023))
    assert [img.get("system:index") for img in collection.images] == ["keep-1", "keep-2"]


def test_mosaic_sentinel2_datatakes_groups_by_system_time_start(monkeypatch) -> None:
    monkeypatch.setattr(gee_scenes, "ee", FakeEE({}))
    collection = FakeCollection(
        [
            FakeImage(
                {
                    "system:index": "20230702T100601_20230702T100844_T32UQD",
                    "system:time_start": 1,
                }
            ),
            FakeImage(
                {
                    "system:index": "20230702T100601_20230702T100844_T33UUU",
                    "system:time_start": 1,
                }
            ),
            FakeImage(
                {
                    "system:index": "20230705T101601_20230705T101730_T32UQD",
                    "system:time_start": 2,
                }
            ),
            FakeImage(
                {
                    "system:index": "20230705T101601_20230705T101730_T33UUU",
                    "system:time_start": 2,
                }
            ),
        ]
    )

    mosaicked = cast(
        FakeCollection,
        gee_scenes._mosaic_sentinel2_datatakes(cast(Any, collection)),
    )

    assert [img.get("system:time_start") for img in mosaicked.images] == [1, 2]
    assert [img.get("scene_id") for img in mosaicked.images] == [
        "20230702T100601",
        "20230705T101601",
    ]
    assert [img.get("system:index") for img in mosaicked.images] == [
        "20230702T100601",
        "20230705T101601",
    ]
