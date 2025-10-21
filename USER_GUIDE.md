# User Guide

## Phenotype Prediction
### Preprocessing
- Genotypic and phenotypic data are read in directly, preprocessed and fed into an RF model
- Metrics such as ROC are calculated for the results

### Input Data
- Genotypic information needs to be provided as derived RGIs CARD results in .gff format
- Phenotypic information needs to be provided as antibiogram in .csv format

### Stacked Random Forest Example
Predict antibiogram for PATRIC samples on prepareddata and trained models:
- Set `MODE = Execution_Mode.PREDICT_ON_SAVED` in the `config.py`
- Run `python workflow/random_forest.py resources/settings/DataPaths_predict.csv`
- See result at `resources/DataSets/BVBRC_use_case/Result_BVBCR_use_case.csv`

### Stacked Random Forest Instructions
- Set `STACK_MODEL = True` in the `config.py`
- Train and Test:
  - Set `EXECUTION_MODE = Execution_Mode.TRAIN_TEST` in the `config.py`
  - Add the paths to the data records that shall be processed to `resources/settings/DataPaths.csv`
  - Run `python workflow/random_forest.py resources/settings/DataPaths.csv`
- Train models and save them for later predictions:
  - Set `EXECUTION_MODE = Execution_Mode.SAVE_TRAINED` in the `config.py`
  - Add the paths to the data records that shall be processed to `resources/settings/DataPaths.csv`
  - Run `python workflow/random_forest.py resources/settings/DataPaths.csv`
- Make predictions with earlier saved models:
  - Set `EXECUTION_MODE = Execution_Mode.PREDICT_ON_SAVED` in the `config.py`
  - Add the paths to the data records that shall be processed to `resources/settings/DataPaths.csv`. The csv corresponding to `PathToCsv` only needs to hold `Sample_ID_IfH` and `Organism_Code`
  - Run `python workflow/random_forest.py resources/settings/DataPaths.csv`

### Stacked Random Forest Architecture
Relies on two RF layers for prediction
- Test and training data sets are split once at the beginning
- Layer 1:
  - Input: pre-processed data
  - For each AB one RF model
  - The models are validated using out-of-fold iterations -> Test data is split into test and validation data
  - Output: Probabilities for each class (S/I/R)
- The Layer 1 results for each AB are combined
- Layer 2:
  - Input: combined input of Layer 1
  - For each AB one RF model
  - Output: The class with the highest probability according to the Layer 2

![alt text](resources/images/Stacked_Diagram.png)
- Reasoning:
- Results of several ABs can be combined with each other despite the RF approach
- Correlations between the AB resistances can be learned
- Combination of multiple ABs is not possible with only one layer, because not all AB resistances are known for any sample and `NaN` is not supported as target value

### One Layer Random Forest
- Set `STACK_MODEL = False` and `EXECUTION_MODE` according to the desired outcome(see stacked RF) in the `config.py`
- Add the paths to the data records that are processed to the `resources/settings/DataPaths.csv`
- The starting point is the class `random_forest.py` which is used as follows:
`python workflow/random_forest.py resources/settings/DataPaths.csv`

### Evaluate Prediction
- Set `EVALUATE_PREDICTION` to true if predicted and real results shall be compared
- The real result csv file needs to contain resistance classification to perform that
- UME test data: `/groups/ds/Win-KID/UKM_Sciebo/Essen_Isolates/subset_0125_UME_Paper`
- UKM test data: `/groups/ds/Win-KID/UKM_Sciebo/Retrospective_Dec22-Jul24_all-species/subset_retro_UKM_Paper`