from __future__ import annotations

import h5py
import numpy as np
import pytest

from sequince.utils.hdf5 import load_from_h5_file, recursively_save_to_h5_file


def test_round_trip_portable_values(tmp_path) -> None:
    path = tmp_path / "values.h5"
    payload = {
        "mapping": {"value": None},
        "list": [True, 3, 2.5, 1 + 2j, "text", b"bytes"],
        "tuple": (np.int16(-2), np.float32(1.5), np.complex64(2j)),
        "arrays": {
            "bool": np.array([True, False]),
            "uint": np.array([1, 2], dtype=np.uint32),
            "float": np.array([[1.0, 2.0]], dtype=np.float64),
            "complex": np.array([1 + 2j], dtype=np.complex128),
            "bytes": np.array([b"a", b"bc"], dtype="S2"),
        },
    }
    with h5py.File(path, "w") as h5_file:
        recursively_save_to_h5_file(h5_file, "nested/results", payload)
    with h5py.File(path, "r") as h5_file:
        loaded = load_from_h5_file(h5_file, "nested/results")

    assert loaded["mapping"] == {"value": None}
    assert loaded["list"] == payload["list"]
    assert isinstance(loaded["tuple"], tuple)
    assert isinstance(loaded["tuple"][0], np.int16)
    for key, expected in payload["arrays"].items():
        assert isinstance(loaded["arrays"][key], np.ndarray)
        np.testing.assert_array_equal(loaded["arrays"][key], expected)


@pytest.mark.parametrize("key", ["", "a/b", ".", "..", "a\x00b", 1])
def test_invalid_mapping_keys_are_rejected(tmp_path, key) -> None:
    with h5py.File(tmp_path / "invalid.h5", "w") as h5_file:
        with pytest.raises((TypeError, ValueError)):
            recursively_save_to_h5_file(h5_file, "value", {key: 1})
        assert "value" not in h5_file


@pytest.mark.parametrize("path", ["", "/absolute", "a//b", "a/../b"])
def test_invalid_paths_are_rejected(tmp_path, path) -> None:
    with h5py.File(tmp_path / "invalid-path.h5", "w") as h5_file:
        with pytest.raises(ValueError):
            recursively_save_to_h5_file(h5_file, path, {"x": 1})


def test_missing_and_existing_paths(tmp_path) -> None:
    with h5py.File(tmp_path / "paths.h5", "w") as h5_file:
        recursively_save_to_h5_file(h5_file, "value", {"x": 1})
        with pytest.raises(FileExistsError):
            recursively_save_to_h5_file(h5_file, "value", {"x": 2})
        with pytest.raises(KeyError):
            load_from_h5_file(h5_file, "missing")


def test_cycles_and_unsupported_values_are_rejected(tmp_path) -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with h5py.File(tmp_path / "unsupported.h5", "w") as h5_file:
        with pytest.raises(ValueError, match="Cyclic"):
            recursively_save_to_h5_file(h5_file, "cycle", {"x": cyclic})
        with pytest.raises(TypeError, match="object"):
            recursively_save_to_h5_file(h5_file, "object", {"x": object()})
        with pytest.raises(TypeError, match="object"):
            recursively_save_to_h5_file(
                h5_file, "dtype", {"x": np.array([object()], dtype=object)}
            )
        with pytest.raises(TypeError, match="<U1"):
            recursively_save_to_h5_file(
                h5_file, "unicode", {"x": np.array(["x"])}
            )


def test_failed_write_removes_incomplete_target(tmp_path) -> None:
    class BrokenArray:
        shape = (1,)
        dtype = np.dtype("f8")

    with h5py.File(tmp_path / "cleanup.h5", "w") as h5_file:
        with pytest.raises(TypeError, match="target/broken"):
            recursively_save_to_h5_file(
                h5_file, "target", {"written": 1, "broken": BrokenArray()}
            )
        assert "target" not in h5_file


def test_malformed_physical_nodes_are_rejected(tmp_path) -> None:
    with h5py.File(tmp_path / "malformed.h5", "w") as h5_file:
        group = h5_file.create_group("value")
        group.attrs["_sequince_hdf5_node_type"] = "list"
        child = group.create_group("00000001")
        child.attrs["_sequince_hdf5_node_type"] = "none"
        with pytest.raises(ValueError, match="non-contiguous"):
            load_from_h5_file(h5_file, "value")
        del h5_file["value"]
        h5_file.create_group("value")
        with pytest.raises(ValueError, match="missing attribute"):
            load_from_h5_file(h5_file, "value")


def test_closed_and_read_only_handles_are_rejected(tmp_path) -> None:
    path = tmp_path / "handles.h5"
    h5_file = h5py.File(path, "w")
    h5_file.close()
    with pytest.raises(ValueError, match="closed"):
        recursively_save_to_h5_file(h5_file, "value", {"x": 1})
    with h5py.File(path, "r") as read_only:
        with pytest.raises(OSError, match="not writable"):
            recursively_save_to_h5_file(read_only, "value", {"x": 1})
