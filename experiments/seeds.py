"""Seed lists for demo vs study."""


def demo_seeds(n: int = 3, base: int = 7):
    return [base + i for i in range(n)]


def study_seeds(n: int = 30, base: int = 100):
    return [base + i for i in range(n)]
