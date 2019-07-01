# self-harm, an ODE parameterisation tool
Based on the generalised profiling / parameter cascading techniques in Ramsay and Hooker's _Dynamic Data Analysis_, this is a tool for parameterising arbitrary ODE systems.

## Usage
```bash
python main.py -a ACTION_TYPE -f RUN_FILE [-o OUTPUT_FILE]
```

### Options
-   `ACTION_TYPE`: `[i]`ntegrate model, `[s]`how data, `[f]`it model to data, or generate `[l]`curve for algorithm parameter selection
-   `RUN_FILE`: Points to a `CONFIG_FILE`, `MODEL_FILE`, `DATA_FILE`s
-   `OUTPUT_FILE`: File to write output to

#### `RUN_FILE` contents

-   `MODEL_FILE`: Contains two functions, `model` and `model_form`.
    -   `model`: (`[ts, xs, ps] -> [dxdt]`) Defines the ODE system
    -   `model_form`: (`None -> dict`) Defines the number of state variables and parameters.
-   `DATA_FILE`: Contains a data set. Can specify multiple data files
-   `CONFIG_FILE`: Specifies algorithm properties
    -   `iv`: Initial Values. Overwritten by data.
    -   `ip`: Initial parameter guess.
    -   `ts`: Time Span. Overwritten by data. In form `start end steps`
    -   `dp`: Function to parse the data with. In form `file_name function_name`
    -   `gs`: Fine grid size for basis function analysis.
    -   `bn`: Number of basis functions
    -   `rf`: Times to refit the model. Currently not used.
    -   `rg`: Regularisation parameter. Fed into error function.
    -   `rv`: Regularisation value. Value of parameters to regularise towards.
    -   `rs`: Resampling parameter. Number of data points to randomly drop when fitting. Currently not used.
    -   `vf`: Data visualisation function.
    -   `kf`: Knot choice function. Custom function to choose knots from data for basis of the smooth.

## Components
-   `ingestor.py`: Ingestion of command line/configuration file contents. Sets up functions to call as specified in configurations.
-   `modeller.py`: Constructs the casadi models that can be differentiated and called for the fitting process
-   `fitter.py`: Constructs the objective functions and minimisation calls that perform the parameter estimation
-   `viewer.py`: The plotting tool for visualising different aspects of the solution

## General Usage Pattern
```python
# Create the context with the ingestor module
import ingestor
context = ingestor.Context(run_file_name="/path/to/run/file")

# Create the solver object (that in turn calls the modeller classes)
import fitter
solver = fitter.Fitter(context=context)

# Run the solver over a number of algorithm parameters, rho
rhos = np.logspace(start=start, stop=stop, num=num)
for rho in rhos:
    solver.solve(rho=rho, propagate=True)

# Inspect solution
import viewer
view = viewer.View(context=context, fitter=solver)
view.show_parameter_values()

# Additional validation options
optimal_rho = view.draw_lcurve(optimal=True)
view.validate_on_confidence()
view.show_iterations()
```

## Auxiliary Function Specifications
### `CONFIG_FILE.dp`
This function should contain the parsing code
It should take in as input the raw data form file, select out the relevant state variable data sets and aggregate them into a dictionary.
the dictionary should look like:
```python
{
  "y": Aggregated data in columns
  "t": Time markers for data in y
  "{key}": A column of data values for plotting
}
```
It can also return an update dictionary that adds or overwrites values in the fitting configuration. Specifically we look for:

-   `weightings` on each state variable for fitting
-   `observation_vector` which defines a mapping between data columns in `"y"` (above) and the state variables in the model

We also wee initial value and integration time span updates as well

### `CONFIG_FILE.kf`
This function should, for a given fine time mesh and the parsed data, return a list of all the knot locations that define the basis for interpolation.
