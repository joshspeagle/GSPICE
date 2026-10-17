import copy
import pickle
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from _golden import same
from gspice import Model

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
    assert same_fields(duplicate(model), model)
    # A model damaged behind its frozen fields fails the checks when it is rebuilt.
    object.__setattr__(model, "n_train", (10, 0, 30))
    with pytest.raises(ValueError, match=r"^n_train\[1\]: "):
        duplicate(model)
