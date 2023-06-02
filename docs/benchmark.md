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

## Execution
### Simple Execution
A benchmark is executed by calling the instance.

```python
experiment = bench()
```

By enabling `autosave`, the benchmark results will be automatically saved in the specified directory. `autosave` is enabled by default.


## Result Analysis
### Obtaining Experiment Results as table
The results of a benchmark execution are returned as an Experiment object. This object contains the solver output results for combination of each parameters.  
The Experiment object can be accessed through its data attribute:

```python
results.table
```

### Changing Solver Output Column Names
The column names for the solver output are based on the name of the solver specified in the constructor. This can be changed using the solver_name parameter:

```python
benchmark = Benchmark(solver, params, solver_name='my_solver')
```

### If the Solver Output is a jijmodeling.SampleSet
In case the output from a solver is a jijmodeling.SampleSet object, each sample, along with its energy and occurrence count, is automatically included in the results table:

```python
```

## Saving and Loading
### Saving Experiment Results
The results of a benchmark can be saved using the save method. The user can choose between saving in JSON or Pickle format:

```python
results.save('my_savedir', format='json')
```
### Saved Directory Structure
In the saved directory, a subdirectory is created for each combination of solver and parameters. This makes it easy to manage a large number of results:

```bash
my_savedir/
    solver1_param1=1_param2=4/
        results.json
    solver1_param1=2_param2=5/
        results.json
    ...
```
This structure can be changed with the savedir parameter:

```python
results.save('my_savedir', savedir='custom_dir')
```
For more detailed explanations and examples for each section, please refer to the detailed description of each section.