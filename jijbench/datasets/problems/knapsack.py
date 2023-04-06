from __future__ import annotations

import jijmodeling as jm


def knapsack():
    w = jm.Placeholder("w", dim=1)
    v = jm.Placeholder("v", dim=1)
    n = jm.Placeholder("n")
    c = jm.Placeholder("c")
    x = jm.Binary("x", shape=(n,))

    # i: itemの添字
    i = jm.Element("i", n)

    problem = jm.Problem("knapsack")

    # objective function
    obj = jm.Sum(i, v[i] * x[i])
    problem += -1 * obj

    # Constraint: knapsack 制約
    const = jm.Constraint("knapsack-constraint", jm.Sum(i, w[i] * x[i]) - c <= 0)
    problem += const

    return problem
