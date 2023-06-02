# Experiment
## Introduction
The Experiment class is a key part of the JijBench. It stores the results obtained from a benchmark in Artifact and Table objects and assists in managing the benchmark process. With this class, you can add and save experimental results, as well as view them in various formats.

This section will walk you through the usage of Experiment class with examples.


## Constructor
### Creating an Instance
To instantiate an Experiment, simply call `Experiment()`:

```python
import jijbench as jb

experiment = jb.Experiment()
```
This creates a new experiment object with a unique ID as the name and the default save directory.

### Custom Naming
If you want to specify the name of your experiment, pass the name as a string:

```python
experiment = Experiment(name="my_experiment")
```

To get the name of the experiment, simply access the name attribute:

```python
experiment.name
```

### Changing a Save Directory
By default, an Experiment object will automatically save data to the default directory defined in the `DEFAULT_RESULT_DIR`. However, you can change the saving directory when instantiating the Experiment object.

```python
experiment = Experiment(savedir="path/to/experiment")
```

## Appending
### Simple Append
The simplest use-case for Experiment is to instantiate an Experiment object and append records to it. Before appending to an experiment, a Record object must be created. A record stores experimental parameters and observed values:

```python
import pandas as pd

# Create a Record
record = jb.Record(pd.Series({"param1": jb.Parameter(1, name="param1"), "param2": jb.Parameter(2, name="param2")}))
```

To add this record to the experiment, use the append() method:

```python
experiment.append(record)
```

## Batch Appending with Loop
Here's an example where we run a hypothetical benchmark for different parameter values and store the results in records, which are then appended to an experiment.

```python
experiment = jb.Experiment(name='my_experiment')

# Hypothetical parameter values
param_values = [1, 2, 3, 4, 5]

for value in param_values:
    # Hypothetical benchmark run with the parameter
    result = run_benchmark(value)

    record = jb.Record(pd.Series({
        "param": jb.Parameter(value, name="param"),
        "result": jb.Response(result, name="result")
    }))

    experiment.append(record)
```

In this example, run_benchmark(value) is a placeholder for your actual benchmark function. It's expected to return a metric value for the given parameter value.

### Appending within a Context Manager
If the context manager is used inside a for loop and the ID is automatically assigned, the code could look something like this:

```python
experiment = jb.Experiment(name='my_experiment')

# Hypothetical parameter values
param_values = [1, 2, 3, 4, 5]

for value in param_values:
    # Use the context manager
    with experiment:
        # Hypothetical benchmark run with the parameter
        result = run_benchmark(value)

        record = jb.Record(pd.Series({
            "param": jb.Parameter(value, name="param"),
            "result": jb.Response(result, name="result")
        }))

        experiment.append(record)

```
In this case, each iteration of the loop will start a new context for the experiment, automatically assigning a new ID for each record. Once the block within the with statement completes, any cleanup or finalization required for the experiment would be automatically performed.

## Data Access
### Accessing a Table
To access the data stored in the experiment as a table, use the `table` property:

```python
experiment.table
```

### Parameters Table
To view only the parameters from the table, use the `params_table` property:

```python
experiment.params_table
```

### Response Table
Similarly, to view only the response data from the table, use the `response_table` property:

```python
experiment.response_table
```

## Saving and Loading
### Automatic Saving
If `autosave=True` when creating an instance, the experiment results will be automatically saved under experiment.savedir. This ensures that every change made to your experiment is captured, preventing data loss due to unexpected program termination or system failure.  
Experiment results are saved under `experiment.savedir`. The directory structure looks like this:
```
experiment.savedir/
|
└───my_experiment/
    |
    ├───artifact/
    |   └───artifact.dill
    |
    └───table/
        └───table.dill
```


### Saving an Experiment
If `autosave=False`, you can manually save the experiment using the `save()` method:

```python
experiment.save()
```

### Loading an Experiment
You can load the experiment results using the `load()` method:

```python
experiment = jb.load('path/to/experiment', experiment_names=['my_experiment'])
```

The first argument of load specifies the directory where the experiment results are saved. Then, by specifying the name of the experiment in the `experiment_names` argument, the results of the experiment can be loaded.  
If you want to load multiple experiment results at the same time, pass multiple names to `experiment_names`.

```python
experiment = jb.load('path/to/experiment', experiment_names=['my_experiment', 'my_experiment2'])
```

This allows you to access and work with the data from multiple experiments simultaneously, facilitating comparison and analysis of results.

