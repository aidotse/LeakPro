"""Dumy test module to test the test suite."""

EXPECTED_RESULT = 2

def test_add() -> None:
    """Test that the add function returns the correct result."""
    a = 1
    b = 1
    assert a + b == EXPECTED_RESULT  # noqa: S101
