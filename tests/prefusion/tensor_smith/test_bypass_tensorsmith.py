import pytest


class Wuhan:
    pass

def test_bypass_tensor_smith():
    from prefusion.dataset.tensor_smith import BypassTensorSmith

    wuhan = Wuhan()
    tensor_smith = BypassTensorSmith()
    res1 = tensor_smith(wuhan)
    assert res1 is wuhan
    number = 123
    res2 = tensor_smith(number)
    assert res2 is number
    assert res2 == number
    alist = [123, 'a']
    res3 = tensor_smith(alist)
    assert res3 is alist
    assert res3 == alist

    with pytest.raises(NotImplementedError):
        _ = tensor_smith.reverse(res3)

def test_bypass_tensor_smith_reg():
    from prefusion.registry import TENSOR_SMITHS
    tensor_smith = dict(type='BypassTensorSmith')

    wuhan = Wuhan()
    tensor_smith = TENSOR_SMITHS.build(tensor_smith)
    res1 = tensor_smith(wuhan)
    assert res1 is wuhan
    number = 123
    res2 = tensor_smith(number)
    assert res2 is number
    assert res2 == number
    alist = [123, 'a']
    res3 = tensor_smith(alist)
    assert res3 is alist
    assert res3 == alist

    with pytest.raises(NotImplementedError):
        _ = tensor_smith.reverse(res3)
