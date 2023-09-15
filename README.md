# Mutation-based Fault Localization of Deep Neural Networks

## Overview

This repository contains source code of `deepmufl`, as well as the dataset of bugs and raw datat used to answer RQ1-2 in the paper.
Below is the directory structure for the repository located at [Box](https://iastate.box.com/s/6thee4ntmma8y7ef3744yw6j9t0i3hfa), due to dataset size.

```plain
.
├── Dataset
│   ├── all-bugs                       # All 109 bugs obtained from StackOverflow and past work (7 from past)
│   ├── raw-so-query.csv               # Original list of 8,412 posts from StackOverflow
│   └── query.sql                      # SQL query used to obtain the initial list of posts
├── src                                # Source code of deepmufl
├── requirements.txt                   # Dependency and Python virutal environment information
└── RQ1-2
    ├── Output Messages                # Output messages of the baseline tools
    ├── DeepDiagnosis Results.csv
    ├── DeepLocalize Results.csv
    ├── Neuralint Results.csv
    ├── UMLAUT Results.csv
    └── deepmufl Results.csv
```

## Dataset
The bugs are stored under the directory `Dataset/all-bugs`.
Each bug is placed in a folder named after the StackOverflow post handle corresponding to it.
The folder name is a unique number which we consider as bug ID.

Under the `Dataset` directory, we have a `.csv` file and a SQL query file.
The file `raw-so-query.csv` contains the result of running the SQL query (`query.sql`) as of December 2022.

## Source Code
To run `deepmufl`, one needs to first create a Conda virtual environment based on the file `requirements.txt`.
We have used [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers) 4.13.0 on 64 Ubuntu Linux (both 18 and 22) to test `deepmufl`.
```shell
conda create --name deepmufl-env --file requirements.txt
```
Once created, one can enter the virtual environment using the following command.
```shell
conda activate deepmufl-env
```
We are now ready to `deepmufl`.
The source code of the tool is located under the directory `src`.
The module `main.py` is the entry point for the program.
One can run the tool as follows.
```shell
python main.py path/to/model.h5 1.0 path/to/inputs.npy path/to/outputs.npy class 0.001
```
The first argument to `main.py` is the `h5` file name for the buggy model.
The second argument is the rate of mutation selection, which is intended to be a real value between 0 and 1.0, with 0.0 meaning no mutation and 1.0 meaning all the generated mutants should be tested 100% 
The next two arguments are the file names for the numpy arrays, stored as `.npy` files, for inputs and outputs.
Next the argument `class` (or `classification`) indicate that the input model is a classifier.
Alternatively we could pass the value `reg` (or `regression`) to signify that the input model is a classifier model.
Lastly, the optional argument `0.001` is the delta value for comparing floating-point values, i.e., two floating-point will be deemed equal if their absolute difference is no more than the delta.
By default this value is set to 1e-3.

## Research Questions
The direcotry `RQ1-2` contains all the information used in RQ 1 and RQ 2 in the paper.
The files `DeepDiagnosis Results.csv`, `DeepLocalize Results.csv`, `Neuralint Results.csv`, and `UMLAUT Results.csv` contain the result of applying the tools DeepDiagnosis, DeepLocalize, Neuralint, and UMLAUT, respectively, on the bugs in our dataset.
These files have three columns: the first column lists the bug ID, the second column reports whether or not the tool detected the bug successfully, and the last column reports the elapsed time in seconds.
For these tools the output messages are reported under the directory `Output Messages`.
Note that in output messages, an "N/A" denotes lack of data, i.e., the tool does not print any output.
Finally, the file `deepmufl Results.csv` reports the results for `deepmufl`.
In this file, various information such as fault localization time, impact of mutation selection (i.e., rows 25%, 50%, 75%, and 100%) on fault localization time, and fault localization effectiveness based on the impact type and formula used.
This file also reports the training time for each model and fault localization time with training time (columns "Total Time (s)").
Similar to the previous files, TRUE denotes bug found, while FALSE denotes bug not found.
