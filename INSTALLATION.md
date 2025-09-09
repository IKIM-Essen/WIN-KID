
This guide explains how to install and set up the **WIN-KID** repository on your local system.

## Prerequisites

Before starting, ensure you have the following installed on your system:

- **Git**  
- **Conda** (or **Miniconda** / **Anaconda**) for environment management  
- Compatible operating system: **MacOS** or **Linux**  
- Tested hardware: Apple M3 with 24 GB RAM  


## Step 1: Clone the Repository

Open a terminal and clone the repository from GitHub:

```bash
git clone https://github.com/IKIM-Essen/WIN-KID.git
cd WIN-KID
```

## Step 2: Set Up the Environment

The project environment is defined in environment.yml. To create and activate it:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate WIN-KID_env
```

## Step 3: Quick Start
Predict antibiogram for PATRIC samples on prepared data and trained models:
- Set `MODE = Execution_Mode.PREDICT_ON_SAVED` in the `config.py`
- Run `path/to/python workflow/random_forest.py resources/settings/DataPaths_predict.csv`
- See result at `/Users/julianzander/Code/WIN-KID/resources/DataSets/BVBRC_use_case/Result_BVBCR_use_case.csv`

## Further Usages

See `USER_GUIDE.md` for an overview of all applications and their instructions

## Dependencies

See `environment.yml` for a list of all used dependencies

