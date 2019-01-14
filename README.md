# self-harm
This is a repository that contains skeleton code for fitting an arbitrary ODE model

## Usage
```
python ./main.py --help
```

| Flag |                                  |
|------|----------------------------------|
|  -a  | Action to take                   |
|  -v  | Write fitting results to console |
|  -vv | As above, and also plot          |
|  -c  | Path to configuration file       |
|  -o  | Path to an output file           |

## Getting Data
Due to the potential size of the data, we do not want to pollute the repository with data files.
We have, however, written a file to automatically retrieve the data files relevant to each model.
```
./gather_sources.sh data/sources.txt
```
This will save the data files in a file named after the datetime you trigger the command.

**Note** that this will only run on \*nix machines (Linux, MacOS).

## Environment Setup
In a virtual environment, you can run:
```
pip install -r requirements.txt
```
to setup the environment for this project

## Just Run It
After sourcing the data, you can automatically run fitting using the `just-run-it` utility script.

```bash
./just-run-it.sh
```

It will prompt for arguments, or they can specified as in the help

```bash
./just-run-it.sh --help
```

## Result Summary Tools
`./view.py` visualises the results of just-run-it as a box graph
`./quick_quantify.py` produces basic summary statistics for a results file

## Analysis Tools
`lcurve.sh` runs an Lcurve analysis
`lcurve.py` visualises the Lcurve analysis results of `lcurve.sh`
