# Benchmark

## Introduction
The Benchmark class is a utility that enables the creation and management of benchmark tests. It allows the user to specify solvers and parameters, and then perform benchmark tests on all combinations of these.

## Constructor
### Definition of Solver and Parameters
The solver and parameters are specified in the Benchmark constructor. Here is an example of how it might be defined:

```python
# Pre-defined solver
def my_solver(param1, param2):
    return param1 + param2

# Parameters
params = {
    "param1": [1, 2, 3],
    "param2": [4, 5, 6],
}
```

### Single Solver Benchmark
Once the parameters and solver have been defined, a benchmark can be created. In case you wish to create a benchmark that only uses a single solver, you can simply pass that solver to the constructor:

```python
import jijbench as jb

bench = Benchmark(solver, params)
```

### Multi-Solver Benchmark
To create a benchmark that uses multiple solvers, you can pass a list of solvers to the constructor:

```python
solvers = [my_solver, my_solver2, my_solver3]
benchmark = jb.Benchmark(solvers, params)
```

### Custom Naming
If you want to specify the name of your benchmark, pass the name as a string:

```python
bench = Benchmark(solver, params, name="my_benchmark")
```

## Execution
### Simple Execution
A benchmark is executed by calling the instance.

```python
experiment = bench()
```

With autosave enabled, which is the default setting, benchmark results automatically save to a specified directory. By default, these results are saved in `DEFAULT_RESULT_DIR` (i.e., ./.jb_results). To change the directory where the experiment results are stored, use the savedir argument:

```python
experiment = bench(savedir='path/to/benchmark')
```

## Result Analysis
### Obtaining Experiment Results as table
The results of a benchmark execution are returned as an Experiment object. This object contains the solver output results for combination of each parameters.  
For a more detailed understanding of Experiments, please refer to the comprehensive documentation provided in the Experiment class description.

### Changing Solver Output Column Names
The column names for the solver output are based on the name of the solver specified in the constructor. This can be changed using the solver_name parameter:

```python
experiment = bench(solver_output_names=["response"])
```

### If the Solver Output is a jijmodeling.SampleSet
If the output from the solver is a jijmodeling.SampleSet object, basic information about the SampleSet, such as energy and objective function values, is automatically expanded into each row of the table:

```python
def my_solve(problem, instance_data) -> jm.SampleSet:
    # Obtain a SampleSet by solving the problem
    return sampleset

bench = Benchmark(my_solver, params)
experiment = bench()

# The SampleSet is automatically expanded into the table
experiment.table["energy"]
experiment.table["objective"]
```

## Saving and Loading
### Saved Directory Structure
If autosave=True during the benchmark run, the experimental results are automatically saved. It is important to note that the directory structure in which the experiment results are saved is slightly different from that of Experiment:

```bash
path/to/benchmark/(i.e., ./.jb_results)
|
└───my_benchmark/
    |
    ├───my_experiment_1/
    |    |
    |    ├───artifact/
    |    |   └───artifact.dill
    |    |
    |    └───table/
    |        └───table.dill
    |
    ├───my_experiment_2/
    |    |
    |    ├───artifact/
    |    |   └───artifact.dill
    |    |
    |    └───table/
    |        └───table.dill
    |
    ...
```

### Loading Experiment Results
To load all the experiment results tied to a benchmark name, use the `load` method and specify the benchmark name as the first argument:

```python
experiment = jb.load("my_benchmark")
```