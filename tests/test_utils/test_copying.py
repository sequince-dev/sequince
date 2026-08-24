import numpy as np
import pytest

from sequince.utils.copying import array_safe_copy


@pytest.mark.parametrize("module_name", ["numpy", "jax.numpy", "torch"])
def test_array_safe_copy_preserves_array_backend(module_name):
    xp = pytest.importorskip(module_name)
    array = xp.asarray([1.0, 2.0])
    value = {"nested": [array], "tuple": (array,)}

    copied = array_safe_copy(value)

    assert copied is not value
    assert copied["nested"] is not value["nested"]
    assert copied["tuple"] is not value["tuple"]
    assert copied["nested"][0] is not array
    assert type(copied["nested"][0]) is type(array)
    np.testing.assert_array_equal(np.asarray(copied["nested"][0]), [1.0, 2.0])


def test_array_safe_copy_handles_non_leaf_torch_tensor():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor(0.0, requires_grad=True).exp()

    copied = array_safe_copy({"tensor": tensor})

    assert torch.equal(copied["tensor"], tensor)
    assert copied["tensor"] is not tensor
