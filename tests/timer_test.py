""""Test the timer context manager."""

from __future__ import annotations

from ragamp.utils import timer


def factorial(n: int) -> int:
    """Dummy function to test the timer context manager."""
    return 1 if n in (0, 1) else n * factorial(n - 1)


with timer('Factorial timing'):
    print(factorial(100))
