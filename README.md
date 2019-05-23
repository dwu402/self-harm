# self-harm, an ODE parameterisation tool
Based on the generalised profiling / parameter cascading techniques in Ramsay and Hooker's _Dynamic Data Analysis_, this is a tool for parameterising arbitrary ODE systems.

## Usage
```bash
python main.py -a ACTION_TYPE -f RUN_FILE [-o OUTPUT_FILE]
```

### Options
-   `ACTION_TYPE`: \[i\]ntegrate model, \[s\]how data, \[f\]it model to data, or generate \[l\]curve for algorithm parameter selection
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
