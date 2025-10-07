
# Installation and Quick Start Guide

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

The project environment is defined in the [environment.yml](./environment.yml). To create and activate it:

```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate WIN-KID_env
```

## Step 3: Quick Start
Predict antibiogram for PATRIC samples on prepared data and trained models:
- Run `python workflow/random_forest.py resources/settings/DataPaths_predict.csv`
- See results in terminal or at `resources/DataSets/BVBRC_use_case/Result_BVBCR_use_case.csv`

## Further Usages

See [USER-GUIDE.md](./USER_GUIDE.md) for an overview of all applications and their instructions

## Dependencies

See [environment.yml](./environment.yml) for a list of all used dependencies
