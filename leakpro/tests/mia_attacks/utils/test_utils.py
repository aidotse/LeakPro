import numpy as np

from leakpro.attacks.utils.utils import softmax_logits

def test_softmax_logits():
    logits = np.array([1, 2, 3])
    sm = softmax_logits(logits)

    assert np.sum(sm) == 1.0
    assert np.all(sm >= 0)
    
    sm_low = softmax_logits(logits, temp = 0.1)
    assert np.sum(sm) == 1.0
    assert np.all(sm >= 0)
    
    sm_high = softmax_logits(logits, temp = 10)
    assert np.sum(sm) == 1.0
    assert np.all(sm >= 0)
    
    assert max(sm) > max(sm_high)
    assert max(sm) < max(sm_low)