from __future__ import annotations

import jijmodeling as jm


def travelling_salesman():
    # 問題
    problem = jm.Problem("travelling-salesman")
    dist = jm.Placeholder("d", ndim=2)
    N = jm.Placeholder("N")

    x = jm.BinaryVar("x", shape=(N, N))
    i = jm.Element("i", belong_to=N)
    j = jm.Element("j", belong_to=N)

    t = jm.Element("t", belong_to=N)

    # Objective Funtion
    sum_list = [t, i, j]
    obj = jm.sum(sum_list, dist[i, j] * x[t, i] * x[(t + 1) % N, j])
    problem += obj

    # const1: onehot for time
    const1 = x[t, :]
    problem += jm.Constraint(
        "onehot-time",
        const1 == 1,
        forall=[
            t,
        ],
    )

    # const2: onehot for location
    const2 = x[:, i]
    problem += jm.Constraint(
        "onehot-location",
        const2 == 1,
        forall=[
            i,
        ],
    )

    return problem
