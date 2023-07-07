from __future__ import annotations

import jijmodeling as jm


def knapsack():
    w = jm.Placeholder("w", ndim=1)
    v = jm.Placeholder("v", ndim=1)
    n = jm.Placeholder("n")
    c = jm.Placeholder("c")
    x = jm.BinaryVar("x", shape=(n,))

    # i: itemの添字
    i = jm.Element("i", belong_to=n)

    problem = jm.Problem("knapsack")

    # objective function
    obj = jm.sum(i, v[i] * x[i])
    problem += -1 * obj

    # Constraint: knapsack 制約
    const = jm.Constraint("knapsack-constraint", jm.sum(i, w[i] * x[i]) - c <= 0)
    problem += const

    return problem
