# self-harm
This is a repository that contains skeleton code for fitting an arbitrary ODE model

## Usage
```
python ./main.py --help
```

| Flag |                        |
|------|------------------------|
|  -f  | Whether or not to fit  |
|  -m  | Path to model file     |
|  -p  | Path to parameter file |
|  -d  | Path to data file      |

## Getting Data
Due to the potential size of the data, we do not want to pollute the repository with data files.
We have, however, written a file to automatically retrieve the data files relevant to each model.
```
./gather_sources.sh
```
This will save the data files in a file named after the datetime you trigger the command.

**Note** that this will only run on *nix machines (Linux, MacOS).

## Environment Setup
In a virtual environment, you can run:
```
pip install -r requirements.txt
```
to setup the environment for this project
