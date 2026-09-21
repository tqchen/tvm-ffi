# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Field metadata and FFI conversion integration for native named tuples."""

from __future__ import annotations

import copy
import inspect
import itertools
from collections import namedtuple
from typing import Any

import pytest
import tvm_ffi
import tvm_ffi.testing
from tvm_ffi import dataclasses as dc
from tvm_ffi.core import Object, TypeSchema
from tvm_ffi.dataclasses import common
from tvm_ffi.dataclasses.common import _make_namedtuple

_counter = itertools.count()


def _key(name: str) -> str:
    return f"testing.namedtuple.{name}_{next(_counter)}"


@pytest.fixture
def record_fields() -> tuple[dc.Field, ...]:
    @dc.py_class(_key("Base"))
    class Base(Object):
        count: int

    @dc.py_class(_key("Measurement"))
    class Measurement(Base):
        scale: float

    return dc.fields(Measurement)


def test_native_tuple_and_metadata(record_fields: tuple[dc.Field, ...]) -> None:
    before = [{slot: getattr(f, slot) for slot in dc.Field.__slots__} for f in record_fields]
    record: Any = _make_namedtuple("MeasurementFields", record_fields)
    value = record(count=2, scale=3)

    assert record.__bases__ == (tuple,)
    assert record._fields == ("count", "scale")
    assert record.__annotations__ == {"count": int, "scale": float}
    assert str(inspect.signature(record)) == "(count, scale)"
    assert tuple(value) == (2, 3.0)
    assert type(value.scale) is float
    assert value[1] == 3.0
    assert value._asdict() == {"count": 2, "scale": 3.0}
    assert repr(value) == "MeasurementFields(count=2, scale=3.0)"
    assert copy.copy(value) == value
    assert record._field_defaults == {}
    assert not dc.is_dataclass(record)
    assert not hasattr(record, "__tvm_ffi_type_info__")
    assert not hasattr(record, "_field_schemas")
    for field, snapshot in zip(record_fields, before):
        assert all(getattr(field, slot) is old for slot, old in snapshot.items())


def test_selected_fields_and_c_class_metadata(record_fields: tuple[dc.Field, ...]) -> None:
    selected: Any = _make_namedtuple("Selected", (record_fields[1], record_fields[0]))
    assert selected._fields == ("scale", "count")
    assert type(selected(1, 2).scale) is float
    assert tuple(selected(1, 2)) == (1.0, 2)

    c_fields = dc.fields(tvm_ffi.testing.TestIntPair)
    c_record: Any = _make_namedtuple("IntPairFields", c_fields)
    assert c_record._fields == ("a", "b")
    assert c_record.__annotations__ == {"a": int, "b": int}
    assert c_record(True, 2) == (1, 2)
    assert type(c_record(True, 2).a) is int


def test_schema_fallback_does_not_change_field(record_fields: tuple[dc.Field, ...]) -> None:
    field = copy.copy(record_fields[1])
    field._ty_schema = None
    record: Any = _make_namedtuple("Fallback", (field,))
    assert type(record(2).scale) is float
    assert field._ty_schema is None
    assert field.type is float


def test_cached_schema_needs_no_annotation(monkeypatch: pytest.MonkeyPatch) -> None:
    field = dc.Field("scale", TypeSchema.from_annotation(float))

    def unexpected_resolution(annotation: Any) -> None:
        pytest.fail("A cached schema must not be resolved again")

    class SchemaResolver:
        from_annotation = staticmethod(unexpected_resolution)

    monkeypatch.setattr(common, "TypeSchema", SchemaResolver)
    record: Any = _make_namedtuple("Cached", (field,))
    assert record.__annotations__ == {}
    assert type(record(2).scale) is float
    assert field.type is None


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda t, x: t(1, x), id="positional"),
        pytest.param(lambda t, x: t(count=1, scale=x), id="keyword"),
        pytest.param(lambda t, x: t(1, scale=x), id="mixed"),
        pytest.param(lambda t, x: t._make(v for v in (1, x)), id="make-generator"),
        pytest.param(lambda t, x: t(1, 0)._replace(scale=x), id="replace"),
        pytest.param(
            lambda t, x: getattr(copy, "replace")(t(1, 0), scale=x),
            id="copy-replace",
            marks=pytest.mark.skipif(not hasattr(copy, "replace"), reason="Python 3.13+"),
        ),
    ],
)
def test_conversion_entry_points(record_fields: tuple[dc.Field, ...], make: Any) -> None:
    record: Any = _make_namedtuple("MeasurementFields", record_fields)

    class FloatInput:
        calls = 0

        def __tvm_ffi_float__(self) -> float:
            self.calls += 1
            return 2.5

    raw = FloatInput()
    value = make(record, raw)
    assert type(value) is record
    assert type(value.scale) is float
    assert value.scale == 2.5
    assert raw.calls == 1
    with pytest.raises(TypeError, match=r"MeasurementFields\.scale:.*expected float") as err:
        make(record, "invalid")
    assert isinstance(err.value.__cause__, TypeError)


def test_container_and_object_conversion() -> None:
    @dc.py_class(_key("Payload"))
    class Payload(Object):
        values: tuple[float, ...]
        mapping: tvm_ffi.Map[str, float]
        object: tvm_ffi.testing.TestIntPair

    record: Any = _make_namedtuple("PayloadFields", dc.fields(Payload))
    pair = tvm_ffi.testing.TestIntPair(1, 2)
    value = record([1, 2], {"x": 3}, pair)
    assert value.values == (1.0, 2.0)
    assert all(type(v) is float for v in value.values)
    assert isinstance(value.mapping, tvm_ffi.Map)
    assert type(value.mapping["x"]) is float
    assert value.mapping["x"] == 3.0
    assert value.object.same_as(pair)


def test_nested_conversion_error() -> None:
    @dc.py_class(_key("Nested"))
    class Nested(Object):
        values: tuple[dict[str, float], ...]

    record: Any = _make_namedtuple("NestedFields", dc.fields(Nested))
    with pytest.raises(TypeError, match=r"NestedFields\.values:") as err:
        record([{"x": "invalid"}])
    assert "element [0]" in str(err.value)
    assert "value for key 'x'" in str(err.value)
    assert "expected float" in str(err.value)
    assert isinstance(err.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "args, kwargs",
    [((), {}), ((1,), {}), ((1, 2, 3), {}), ((1, 2), {"scale": 3}), ((1, 2), {"other": 3})],
)
def test_native_constructor_errors(
    record_fields: tuple[dc.Field, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    record: Any = _make_namedtuple("MeasurementFields", record_fields)
    with pytest.raises(TypeError):
        record(*args, **kwargs)


def test_native_helper_errors(record_fields: tuple[dc.Field, ...]) -> None:
    record: Any = _make_namedtuple("MeasurementFields", record_fields)
    native: Any = namedtuple(record.__name__, record._fields)
    # Python versions differ in the native exception type; retain that behavior.
    for values in ([1], [1, 2, 3]):
        with pytest.raises((TypeError, ValueError)) as expected:
            native._make(values)
        with pytest.raises(type(expected.value)) as actual:
            record._make(iter(values))
        assert str(actual.value) == str(expected.value)
    with pytest.raises((TypeError, ValueError)) as expected:
        native(1, 2)._replace(other=3)
    with pytest.raises(type(expected.value)) as actual:
        record(1, 2)._replace(other=3)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("names", [("_private",), ("repeat", "repeat"), ("for",)])
def test_native_field_name_errors(names: tuple[str, ...]) -> None:
    fields = tuple(dc.Field(name, TypeSchema.from_annotation(int)) for name in names)
    with pytest.raises(ValueError):
        _make_namedtuple("Invalid", fields)


def test_empty_and_cls_field() -> None:
    empty: Any = _make_namedtuple("Empty", ())
    assert empty() == empty._make(iter(())) == empty()._replace() == ()
    assert empty._fields == ()
    record: Any = _make_namedtuple(
        "ClassField", (dc.Field("cls", TypeSchema.from_annotation(float)),)
    )
    assert type(record(cls=3).cls) is float
    assert record(cls=3).cls == 3.0


def test_source_initialization_is_unused() -> None:
    def forbidden() -> Any:
        pytest.fail("Tuple creation must not run source initialization")

    @dc.py_class(_key("Source"))
    class Source(Object):
        count: int = dc.field(default=7, kw_only=True)
        values: tuple[int, ...] = dc.field(default_factory=forbidden)
        hidden: int = dc.field(default=9, init=False)

        def __init__(self) -> None:
            forbidden()

        def __post_init__(self) -> None:
            forbidden()

        @dc.init_property
        def computed(self) -> int:
            return forbidden()

    record: Any = _make_namedtuple("SourceFields", dc.fields(Source))
    assert record._fields == ("count", "values", "hidden", "computed")
    assert record(1, [2], 3, 4) == (1, (2,), 3, 4)
    assert record._make([1, [2], 3, 4])._replace(count=5) == (5, (2,), 3, 4)
    for length in range(4):
        with pytest.raises(TypeError):
            record(*(1, [2], 3, 4)[:length])


def test_subclass_helpers(record_fields: tuple[dc.Field, ...]) -> None:
    record: Any = _make_namedtuple("MeasurementFields", record_fields)

    class Subclass(record):
        __slots__ = ()

    value = Subclass._make([1, 2])._replace(scale=3)
    assert type(value) is Subclass
    assert type(value.scale) is float
    assert value.scale == 3.0
