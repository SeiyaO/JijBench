from __future__ import annotations

from dataclasses import dataclass, is_dataclass

import numpy as np
import pandas as pd
import pytest

import jijbench as jb


@dataclass
class SamplerParameters:
    int_param: int
    float_param: float
    str_param: str
    bool_param: bool
    list_param: list
    dict_param: dict


def test_record():
    inputs = [
        jb.ID(name="id1"),
        jb.Date(name="date1"),
        jb.Array(np.arange(5), name="array1"),
    ]
    data = pd.Series(inputs)
    jb.Record(data)


def test_artifact():
    from jijbench.typing import ArtifactDataType

    data: ArtifactDataType = {
        "0": {
            "id": jb.ID(name="id1"),
            "date": jb.Date(name="date1"),
            "array": jb.Array(np.arange(5), name="array1"),
        },
        "1": {
            "id": jb.ID(name="id2"),
            "date": jb.Date(name="date2"),
            "array": jb.Array(np.arange(5), name="array2"),
        },
    }
    jb.Artifact(data)


def test_table():
    inputs = [
        [
            jb.ID(name="id1"),
            jb.Date(name="date1"),
            jb.Array(np.arange(5), name="array1"),
        ]
    ]
    data = pd.DataFrame(inputs)
    jb.Table(data)


def test_record_invalid_data():
    with pytest.raises(TypeError):
        jb.Record(1)

    with pytest.raises(TypeError):
        inputs = [
            "id",
            pd.Timestamp.now(),
            np.arange(5),
        ]
        data = pd.Series(inputs)
        jb.Record(data)


def test_artifact_invalid_data():
    with pytest.raises(TypeError):
        jb.Artifact(1)

    with pytest.raises(TypeError):
        data = {
            "0": {
                "id": "id",
                "date": pd.Timestamp.now(),
                "array": np.arange(5),
            }
        }
        jb.Artifact(data)


def test_table_invalid_data():
    with pytest.raises(TypeError):
        jb.Table(1)

    with pytest.raises(TypeError):
        inputs = [
            [
                "id1",
                pd.Timestamp.now(),
                np.arange(5),
            ]
        ]
        data = pd.DataFrame(inputs)
        jb.Table(data)


def test_record_append():
    factory = jb.functions.RecordFactory()

    inputs1 = [
        jb.ID(name="id1"),
        jb.Date(name="date1"),
        jb.Array(np.arange(5), name="array1"),
    ]
    inputs2 = [
        jb.ID(name="id2"),
        jb.Date(name="date2"),
        jb.Array(np.arange(5), name="array2"),
    ]

    name = "r1"
    r1 = factory(inputs1, name=name)
    r2 = factory(inputs2, name="r2")

    r1.append(r2)

    assert isinstance(r1, jb.Record)
    assert r1.name == name
    assert "id2" in r1.data.index
    assert "date2" in r1.data.index
    assert "array2" in r1.data.index
    assert r1.operator is not None
    assert len(r1.operator.inputs) == 2
    for i in ["id2", "date2", "array2"]:
        assert r1.data[i] == r2.data[i]


def test_table_append():
    factory = jb.functions.TableFactory()

    name = "t1"
    data = [jb.ID("id"), jb.Date(), jb.Array(np.arange(5), "array")]
    r1 = jb.Record(pd.Series(data), "a")
    table = factory([r1], name=name)

    r2 = jb.Record(pd.Series(data), "b")
    table.append(r2)

    assert isinstance(table, jb.Table)
    assert table.name == name
    assert table.data.index[0] == "a"
    assert table.data.index[1] == "b"
    assert table.operator is not None
    assert len(table.operator.inputs) == 2
    for i, d in enumerate(table.data.loc["a"]):
        assert d == table.operator.inputs[0].data.loc["a", i]


def test_artifact_append():
    factory = jb.functions.ArtifactFactory()

    name = "a1"
    data = [jb.ID("id"), jb.Date(), jb.Array(np.arange(5), "array")]
    r1 = jb.Record(pd.Series(data), "a")
    artifact = factory([r1], name=name)

    r2 = jb.Record(pd.Series(data), "b")
    artifact.append(r2)

    assert isinstance(artifact, jb.Artifact)
    assert artifact.name == name
    assert "a" in artifact.data
    assert "b" in artifact.data
    assert artifact.operator is not None
    assert len(artifact.operator.inputs) == 2
    for i, d in artifact.data["a"].items():
        assert d == artifact.operator.inputs[0].data["a"][i]


def test_experiment_append():
    ename = "test"
    e = jb.Experiment(name=ename)

    for i in range(3):
        with e:
            data = [
                jb.Date(),
                jb.Number(i, "num"),
                jb.Array(np.arange(5), "array"),
            ]
            record = jb.functions.RecordFactory()(data)
            e.append(record)

    assert isinstance(e, jb.Experiment)
    assert e.operator is not None
    assert e.data[0].operator is not None
    assert e.data[1].operator is not None
    assert len(e.operator.inputs) == 2

    for i, index in zip(range(3), e.artifact):
        assert e.artifact[index]["num"] == i
        assert e.table.loc[index, "num"] == i


def test_table_view():
    parameters = SamplerParameters(
        int_param=1,
        float_param=2.0,
        str_param="a",
        bool_param=True,
        list_param=[1, 2, 3],
        dict_param={"a": 1, "b": 2},
    )
    inputs = [
        jb.ID(name="id1"),
        jb.Date(name="date1"),
        jb.Array(np.arange(5), name="array1"),
        jb.Parameter({"a": 1}, name="param1"),
        jb.Parameter({"b": 2}, name="param2"),
        jb.Parameter(parameters, name="param3"),
    ]

    record = jb.functions.RecordFactory()(inputs)
    table = jb.functions.TableFactory()([record])

    df = table.view()
    assert "id1" in df.columns
    assert "param1[a]" in df.columns
    assert "param2[b]" in df.columns

    assert df["param1[a]"][0] == 1
    assert df["param2[b]"][0] == 2

    # Do not keep the original dataclass
    assert "param3" not in df.columns

    # Expand dataclass
    assert df["param3.int_param"][0] == 1
    assert df["param3.float_param"][0] == 2.0
    assert df["param3.str_param"][0] == "a"
    assert df["param3.bool_param"][0] == np.bool_(True)
    assert df["param3.list_param"][0] == [1, 2, 3]
    # Do not expand dictionaries in dataclass
    assert df["param3.dict_param"][0] == {"a": 1, "b": 2}
