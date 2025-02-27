# WIN-KID

## How To Use
- names.csv : List for uniform Oganism Name & Code and the Eucast json it should use
- translations.csv : Manual translation of typos / different vitek names for mic interpretation
- ignore.csv : Antibiotika Names that should be ignored in no match handling (interpreter)
1. Run vitek_parser_ukm.py for UKM VITEK data, run vitek_parser_ume.py for UME VITEK data
1.1 Run via: python vitek_parser_ume.py directory/vitek.csv directory/output/files/
1.2 Run via: python vitek_parser_ukm.py directory/vitek.csv directory/output/files/file.csv
2. Run mic_interpreter.py to interpret any parsed files
2.1 Run via: python mic_interpreter.py directory/input/files directory/output/files/

## Parser UME
- works with tab seperated csv files only
- Splits the Vitek file by bacteria name
- Removes unnecessary data and restructures the file
- specify output FOLDER for the parsed vitek files

## Parser UKM
- Removes unnecessary columns
- specify output FILE for the parsed vitek file

## Interpreter
- Can handle single and multiple Input files and can handle multiple organisms in one file
- Antibiotics from a vitek file that can not be matched to an EUCAST antibiotic are skipped and logged via print
- For a match, the name of the VITEK antibiotic must be found in full in the EUCAST table
- If several values are found, the one with the highest similarity score is used. If two values have the same similarity, any one is used.
- If the selected value falls below a certain similarity score, it is also discarded and logged via print.
- If the EUCAST values for "S <=" and "R >" are different:
  - any VITEK value below or equal to the "S <=" EUCAST Value will be interpreted as "S"
  - any VITEK value between the "S <=" and "R >" EUCAST Value will be interpreted as "I"
  - any VITEK value above the "R >" EUCAST Value will be interpreted as "R"
- If the EUCAST values for "S <=" and "R >" are the same:
  - any VITEK value containing ">" will be interpreted as "R"
  - everything else will be interpteted as "S

## ToDo
- Use different cartridges
- Use different species
- Use different VITEK sources (UKM)
- Differentiate between oral and non oral?
- What shall happen with EUCAST values that are doublets (e.g.: due extra information)

## Restrictions
- Only E.coli
- Only AST-N428 cartridge
- Only UKM VITEK files