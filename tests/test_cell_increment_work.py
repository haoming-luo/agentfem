import numpy as np
import pytest

from agentfem.elements._bulk_viscosity import _cell_increment_work


@pytest.mark.parametrize("chunk", [1, 7, 4096, 10000])
def test_bounded_work_retains_shared_nodes_and_conjugacy(chunk):
    rng = np.random.default_rng(76)
    nodes = rng.integers(0, 83, (5001,8))
    force = rng.normal(size=(5001,8,3))
    increment = rng.normal(size=(83,3))
    saved = increment.copy()
    products = force * increment[nodes]
    work, absolute = _cell_increment_work(force, nodes, increment, chunk_size=chunk)
    np.testing.assert_allclose(work, products.sum(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(absolute, np.abs(products).sum(), rtol=1e-12)
    np.testing.assert_array_equal(increment, saved)


@pytest.mark.parametrize("kind", ["nan", "bad_index", "complex", "chunk"])
def test_bounded_work_rejects_invalid_inputs(kind):
    force = np.ones((1,8,3)); nodes = np.arange(8)[None]
    increment = np.ones((8,3)); options = {}
    if kind == "nan": force[0,0,0] = np.nan
    if kind == "bad_index": nodes[0,0] = -1
    if kind == "complex": force = force.astype(complex)
    if kind == "chunk": options["chunk_size"] = 0
    with pytest.raises(ValueError):
        _cell_increment_work(force, nodes, increment, **options)


def test_empty_work():
    assert _cell_increment_work(np.zeros((0,8,3)), np.zeros((0,8), dtype=int),
                                np.zeros((0,3))) == (0., 0.)
