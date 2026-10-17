import copy
import pickle
import re
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from _golden import same
from gspice import Model, __version__, load, save

NBIN, NPIX = 3, 4


def arguments():
    return {
        "mean": np.zeros((NBIN, NPIX)),
        "cov": tuple(k * np.eye(NPIX) for k in (1.0, 2.0, 3.0)),
        "n_train": (10, 20, 30),
        "edges": (1.0, 2.0),
    }


def with_cov(matrix):
    cov = list(arguments()["cov"])
    cov[1] = matrix
    return cov


def same_fields(a, b):
    return all(
        same(getattr(a, f.name), getattr(b, f.name)) for f in fields(Model) if f.init
    )


# The entry [2, 0] is one ulp above its mirror entry [0, 2].
ASYMMETRIC = np.full((NPIX, NPIX), 0.5) + np.eye(NPIX)
ASYMMETRIC[2, 0] = np.nextafter(ASYMMETRIC[2, 0], 1.0)
# Symmetric, but a NaN is not equal to itself, so the symmetry check would reject it
# too: only the message shows that the finite check runs first.
NAN_MATRIX = np.eye(NPIX)
NAN_MATRIX[0, 3] = NAN_MATRIX[3, 0] = np.nan


@pytest.mark.parametrize(
    ("field", "value", "error", "message"),
    [
        ("mean", np.zeros(NPIX), ValueError, r"^mean: "),
        ("mean", np.zeros((0, NPIX)), ValueError, r"^mean: "),
        ("mean", np.zeros((NBIN, 0)), ValueError, r"^mean: "),
        ("mean", np.full((NBIN, NPIX), np.nan), ValueError, r"^mean: "),
        ("mean", np.zeros((NBIN, NPIX), complex), TypeError, r"^mean: "),
        ("mean", [[0.0] * NPIX] * (NBIN - 1) + [[0.0]], ValueError, r"^mean: "),
        ("cov", arguments()["cov"][:2], ValueError, r"^cov: "),
        ("cov", with_cov(np.eye(NPIX - 1)), ValueError, r"^cov\[1\]: "),
        ("cov", with_cov(NAN_MATRIX), ValueError, r"^cov\[1\]: expected finite"),
        ("cov", with_cov(ASYMMETRIC), ValueError, r"^cov\[1\]: "),
        ("cov", with_cov(np.diag([1.0, 0.0, 1.0, 1.0])), ValueError, r"^cov\[1\]: "),
        ("cov", with_cov(np.eye(NPIX, dtype=complex)), TypeError, r"^cov\[1\]: "),
        ("cov", 5, TypeError, r"^cov: "),
        ("n_train", (10, 20), ValueError, r"^n_train: "),
        ("n_train", (10, 0, 30), ValueError, r"^n_train\[1\]: "),
        ("n_train", (10.0, 20, 30), TypeError, r"^n_train\[0\]: "),
        ("n_train", (10, 20, True), TypeError, r"^n_train\[2\]: "),
        ("n_train", (np.True_, 20, 30), TypeError, r"^n_train\[0\]: "),
        ("n_train", 5, TypeError, r"^n_train: "),
        ("edges", (1.0,), ValueError, r"^edges: "),
        ("edges", (1.0, np.nan), ValueError, r"^edges\[1\]: "),
        ("edges", (1.0, 1.0), ValueError, r"^edges: "),
        ("edges", (2.0, 1.0), ValueError, r"^edges: "),
        ("edges", "12", TypeError, r"^edges\[0\]: "),
        ("center", np.nan, ValueError, r"^center: "),
        ("center", True, TypeError, r"^center: "),
        ("weighted", 1, TypeError, r"^weighted: "),
        ("weighted", "yes", TypeError, r"^weighted: "),
    ],
)
def test_an_invalid_field_is_rejected_with_a_message_that_starts_with_its_name(
    field, value, error, message
):
    with pytest.raises(error, match=message):
        Model(**(arguments() | {field: value}))


@pytest.mark.parametrize("container", [list, np.stack])
def test_a_model_stores_float64_c_arrays_and_python_scalars(container):
    mean = np.arange(NBIN * NPIX, dtype=np.float32).reshape(NBIN, NPIX)
    cov = [np.asfortranarray(k * np.eye(NPIX, dtype=np.float32)) for k in (1, 2, 3)]
    model = Model(
        np.asfortranarray(mean),
        container(cov),
        [np.int32(5), np.uint64(6), 7],
        edges=[np.float32(0.5), 1],
        center=np.float32(0.25),
        weighted=np.True_,
    )
    assert same(model.mean, mean.astype(np.float64))
    assert same(model.cov, arguments()["cov"])
    assert model.mean.flags.c_contiguous
    assert all(C.flags.c_contiguous for C in model.cov)
    assert (model.n_train, model.edges, model.center) == ((5, 6, 7), (0.5, 1.0), 0.25)
    assert {type(n) for n in model.n_train} == {int}
    assert {type(e) for e in model.edges} == {float}
    assert type(model.center) is float
    assert model.weighted is True


def test_a_model_holds_read_only_copies_and_leaves_its_inputs_alone():
    args = arguments()
    inputs = [args["mean"], *args["cov"]]
    model = Model(**args)
    for array, original in zip([model.mean, *model.cov], inputs, strict=True):
        assert not array.flags.writeable
        assert original.flags.writeable
        assert not np.shares_memory(array, original)


def test_a_model_is_frozen_and_equals_only_itself():
    model, other = Model(**arguments()), Model(**arguments())
    with pytest.raises(FrozenInstanceError):
        model.center = 1.0
    assert model != other
    assert len({model, other}) == 2


def test_edges_center_and_weighted_are_keyword_only():
    with pytest.raises(TypeError, match="positional"):
        Model(*arguments().values())


DUPLICATES = {
    "pickle": lambda m: pickle.loads(pickle.dumps(m)),
    "copy": copy.copy,
    "deepcopy": copy.deepcopy,
}


@pytest.mark.parametrize("duplicate", DUPLICATES.values(), ids=list(DUPLICATES))
def test_pickle_and_copy_rebuild_a_model_through_its_checks(duplicate):
    model = Model(**arguments(), center=0.95, weighted=True)
    model._precision(0)
    copy = duplicate(model)
    assert same_fields(copy, model)
    # Without the cache of the original.
    assert copy._cache == {}
    # A model damaged behind its frozen fields fails the checks when it is rebuilt.
    object.__setattr__(model, "n_train", (10, 0, 30))
    with pytest.raises(ValueError, match=r"^n_train\[1\]: "):
        duplicate(model)


@st.composite
def models(draw):
    nbin = draw(st.integers(1, 3))
    npix = draw(st.integers(1, 6))
    values = st.floats(-1e100, 1e100)
    positive = st.floats(1e-100, 1e100)
    mean = draw(hnp.arrays(np.float64, (nbin, npix), elements=values))
    cov = []
    for _ in range(nbin):
        A = draw(hnp.arrays(np.float64, (npix, npix), elements=values))
        # A + A.T is exactly symmetric, because float addition commutes.
        C = (A + A.T) / 2
        np.fill_diagonal(C, draw(hnp.arrays(np.float64, npix, elements=positive)))
        cov.append(C)
    n_train = draw(st.lists(st.integers(1, 10**6), min_size=nbin, max_size=nbin))
    edges = st.sets(st.floats(-1e6, 1e6), min_size=nbin - 1, max_size=nbin - 1)
    center = draw(st.none() | values)
    return Model(
        mean,
        cov,
        n_train,
        edges=sorted(draw(edges)),
        center=center,
        weighted=draw(st.booleans()),
    )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(model=models())
def test_load_of_save_gives_bit_identical_arrays_and_equal_scalars(tmp_path, model):
    save(model, tmp_path / "model.npz")
    assert same_fields(load(tmp_path / "model.npz"), model)


@pytest.mark.parametrize("kind", [str, Path])
def test_save_writes_exactly_the_given_path_for_a_str_and_a_path(tmp_path, kind):
    model = Model(**arguments())
    path = kind(tmp_path / "model.dat")
    save(model, path)
    assert [p.name for p in tmp_path.iterdir()] == ["model.dat"]
    assert same_fields(load(path), model)


def test_save_writes_the_keys_and_dtypes_of_the_format(tmp_path):
    save(Model(np.zeros((1, 2)), (np.eye(2),), (5,)), tmp_path / "model.npz")
    with np.load(tmp_path / "model.npz") as archive:
        stored = dict(archive)
    version = stored.pop("gspice_version")
    assert (version.item(), version.shape) == (__version__, ())
    assert {key: (value.dtype, value.shape) for key, value in stored.items()} == {
        "schema": (np.int64, ()),
        "mean": (np.float64, (1, 2)),
        "n_train": (np.int64, (1,)),
        "edges": (np.float64, (0,)),
        "center": (np.float64, ()),
        "weighted": (np.bool_, ()),
        "cov0": (np.float64, (2, 2)),
    }
    assert stored["schema"] == 1
    assert np.isnan(stored["center"])


@pytest.mark.parametrize("schema", [None, 2, 1.0, True, np.array([1])])
def test_load_rejects_an_archive_whose_schema_is_not_the_integer_1(tmp_path, schema):
    path = tmp_path / "model.npz"
    save(Model(**arguments()), path)
    with np.load(path) as archive:
        contents = dict(archive)
    del contents["schema"]
    if schema is not None:
        contents["schema"] = schema
    np.savez(path, **contents)
    with pytest.raises(ValueError, match=rf"^path: .* in {re.escape(str(path))}$"):
        load(path)


@pytest.mark.parametrize(
    ("extra", "drop", "message"),
    [
        ({"noise": np.True_}, (), r"^path: .* got noise, written by gspice .*, in "),
        # The arrays of a low-rank bin in place of its covariance.
        (
            {
                "vectors0": np.eye(NPIX),
                "values0": np.ones(NPIX),
                "diag0": np.ones(NPIX),
            },
            ("cov0", "cov1", "cov2"),
            r"^path: .* got diag0, values0, vectors0, ",
        ),
        ({}, ("center",), r"^path: .* got no center in "),
        ({"cov3": np.eye(NPIX)}, (), r"^cov: "),
    ],
)
def test_load_rejects_a_file_whose_arrays_differ_from_those_of_save(
    tmp_path, extra, drop, message
):
    path = tmp_path / "model.npz"
    save(Model(**arguments()), path)
    with np.load(path) as archive:
        contents = {key: archive[key] for key in archive.files if key not in drop}
    np.savez(path, **(contents | extra))
    with pytest.raises(ValueError, match=message):
        load(path)


def test_load_raises_os_error_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load(tmp_path / "missing.npz")


@pytest.mark.parametrize("damage", ["truncate", "flip"])
def test_load_raises_value_error_for_a_damaged_archive(tmp_path, damage):
    path = tmp_path / "model.npz"
    model = Model(**arguments())
    save(model, path)
    data = bytearray(path.read_bytes())
    if damage == "truncate":
        data = data[:100]
    else:
        # A bit of the data of cov0, which the CRC-32 of its member covers.
        data[data.find(model.cov[0].tobytes()) + 8] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match=r"^path: .* damaged zip archive"):
        load(path)


def test_save_and_load_reject_an_integer_path():
    # open would take an integer as a file descriptor.
    with pytest.raises(TypeError):
        save(Model(**arguments()), 10**6)
    with pytest.raises(TypeError):
        load(10**6)


def test_save_rejects_what_is_not_a_model(tmp_path):
    # fit returns the model and the training mask.
    with pytest.raises(TypeError, match="^model: "):
        save((Model(**arguments()), None), tmp_path / "model.npz")


@pytest.mark.parametrize("name", ["model.npy", "model.txt"])
def test_load_rejects_a_file_that_is_not_an_npz_archive(tmp_path, name):
    path = tmp_path / name
    if name.endswith(".npy"):
        np.save(path, np.zeros(3))
    else:
        path.write_text("1\n")
    with pytest.raises(ValueError, match=r"^path: .* not a zip archive in "):
        load(path)
