from leakpro.leakpro import (
    LEAKPRO,
)  # look in leakpro package for a module named leakpro.py and import LEAKPRO class


def test_add():
    a = LEAKPRO()
    assert a.add(2, 3) == 5
