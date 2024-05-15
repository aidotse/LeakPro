"""General utils functions."""
from typing import Union


def assert_x_in_bound(*,
    x: Union[float, int],
    x_name: str = "",
    low_bound: float = 0.0,
    high_bound: float = 1.0,
    inclusive_flag: bool = False
) -> Union[None, ValueError]:
    """Auxiliar function to assert x is between low_bound and high_bound.

    If x not between bounds, raises ValueError

    Parameters
    ----------
    x : float
        Value to check if between bounds.
    x_name : str, default is ''
        Name of parameter for ValueError message
    low_bound : float, default is 0.0
        Lower bound.
    high_bound : float, default is 1.0
        Higher bound.
    inclusive_flag : bool, default is False
        If True, x can be equal to low_bound/high_bound (ie interval is closed []).

    """
    if len(x_name)>0:
        x_name = f" `{x_name}`"
    if inclusive_flag:
        cond = x < low_bound or x > high_bound
        extra = "="
    else:
        cond = x <= low_bound or x >= high_bound
        extra = ""
    if cond:
        raise ValueError(f"Parameter{x_name} must be >{extra} {low_bound} and <{extra} {high_bound}. Got {x} instead.")
