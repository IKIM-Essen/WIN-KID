# WIN-KID

## Parser UME

- Run via: `python workflow/vitek_parser_ume.py directory/vitek.csv directory/output/` with test data at `resources/test_data`
- works with tab seperated csv files only
- Removes unnecessary columns
- specify output FILE for the parsed vitek file

## Parser UKM

- Run via: `python workflow/vitek_parser_ukm.py directory/vitek.csv directory/output/file.csv` with test data at `resources/test_data`
- Splits the Vitek file by bacteria name and replaces them with bactia code (see names.csv)
- Removes unnecessary data and restructures the file
- specify output FOLDER for the parsed vitek files
- works with ";" seperated csv files only

## Parser BVBCR

- Run via: `python workflow/vitek_parser_bvbcr.py directory/vitek.csv directory/output/file.csv` with test data at `resources/test_data`
- Splits the Vitek file by bacteria name and replaces them with bactia code (see names.csv)
- Removes unnecessary data and restructures the file
- specify output FOLDER for the parsed vitek files
- works with tab seperated csv files only

## Interpreter

- Run mic_interpreter.py to interpret any parsed files:
  `python workflow/mic_interpreter.py directory/input/files directory/output/files/`
- Can handle single and multiple Input files and can handle multiple organisms in one file
- Antibiotics from a vitek file that can not be matched to an EUCAST antibiotic are skipped and logged via print
- Antibiotics with missing EUCAST Values (e.g.: '"S <=": "-",') are removed from the interpreted table
- For a match, the name of the VITEK antibiotic must be found in full in the EUCAST table
- If several values are found, the one with the highest similarity score is used. If two values have the same similarity, any one is used.
- If the selected value falls below a certain similarity score, it is also discarded and logged via print.
- If the EUCAST values for "S <=" and "R >" are different:
  - any VITEK value below or equal to the "S <=" EUCAST Value will be interpreted as "S"
  - any VITEK value between the "S <=" and "R >" EUCAST Value will be interpreted as "I"
  - any VITEK value above the "R >" EUCAST Value will be interpreted as "R"
- If the EUCAST values for "S <=" and "R >" are the same:
  - any VITEK value containing ">" will be interpreted as "R"
  - everything else will be interpteted as "S"

## GFF3 formatter

- Run via: `python gff3_formatter.py /path/to/input_folder /path/to/output_folder`
- Transforms card annotation txt files in gff3 files

## Settings

- `names.csv` : List for uniform Oganism Name & Code and the Eucast json it should use
- `translations.csv` : Manual translation of typos / different vitek names for mic interpretation
- `ignore.csv` : Antibiotika Names that should be ignored in no match handling (interpreter)

## Preprocessing and random forest
- Genotypic and phenotypic data are read in directly, preprocessed and fed into an RF model
- Metrics such as ROC are calculated for the results
- The starting point is the class `random_forest.py` which is used as follows:
`path/to/python workflow/random_forest.py /groups/ds/Win-KID/UKM_Subset_300/interpreted_retrospective_MHK_ECO_428_371.csv  /groups/ds/Win-KID/UKM_Subset_300/card_gff3`

## ToDo
- Random forest results shall also be calculated I and S. Currently only R is used.

- Use different cartridges
- Use different species
- Differentiate between oral and non oral?
- What shall happen with EUCAST values that are doublets (e.g.: due extra information)
- long term solution for phenotype data NaNs (currently -> S)
- long term solution for genotype data NaNs (currently -> 0)