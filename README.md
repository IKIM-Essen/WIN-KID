# WIN-KID

## How To Use
1. Run vitek_parser.py to generate a vitek_parsed.csv from a UKM VITEK File
2. Run mic_interpreter.py to generate a mic_interpretation.csv from a vitek_parsed.csv

## Parser
- Removes all columns of the UKM file except for Sample_ID_IfH ,Organism_Code ,Card_Name and the antibiotic MHK values

## Interpreter
- Antibiotics from a vitek file that can not be matched to an EUCAST antibiotic are skipped and logged via print
- For a match, the name of the VITEK antibiotic must be found in full in the EUCAST table
- If several values are found, the one with the highest similarity score is used. If two values have the same similarity, any one is used.
- If the selected value falls below a certain similarity score, it is also discarded and logged via print.
- All VITEK values with "<" are automatically "S"
- The ">" "=" are removed from the VITEK files and directly compared with the EUCAST values

## Preprocessing and random forest
- Genotypic and phenotypic data are read in directly, preprocessed and fed into an RF model
- Metrics such as ROC are calculated for the results
- The starting point is the class `random_forest.py` which is used as follows:
`path/to/python WIN-KID/wokflow/random_forest.py output/mic_interpretation.csv resources/genotype`

## ToDo
- Random forest results shall also be calculated I and S. Currently only R is used.
- Use different cartridges
- Use different species
- Use different VITEK sources (UKM)
- Differentiate between oral and non oral?
- What shall happen with EUCAST values that are doublets (e.g.: due extra information)

## Restrictions
- Only E.coli
- Only AST-N428 cartridge
- Only UKM VITEK files