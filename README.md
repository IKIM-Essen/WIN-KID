# WIN-KID

## How To Use
- names.csv : List for uniform Oganism Name & Code and the Eucast json it should use
- translations.csv : Manual translation of typos / different vitek names for mic interpretation
- ignore.csv : Antibiotika Names that should be ignored in no match handling (interpreter)
1. Run vitek_parser_new.py to generate multiple {bacteria_name}.csv files in output/vitek_parsed from UKM VITEK data
2. Run mic_interpreter.py to generate multiple {bacteria_name}.csv files in output/interpreted with the MIC Interpretation from the Eucast json files

## Parser
- Splits the Vitek file by bacteria name
- Removes unnecessary columns and restructures the file

## Interpreter
- Antibiotics from a vitek file that can not be matched to an EUCAST antibiotic are skipped and logged via print
- For a match, the name of the VITEK antibiotic must be found in full in the EUCAST table
- If several values are found, the one with the highest similarity score is used. If two values have the same similarity, any one is used.
- If the selected value falls below a certain similarity score, it is also discarded and logged via print.
- All VITEK values with "<" are automatically "S"
- The ">" "=" are removed from the VITEK files and directly compared with the EUCAST values

## ToDo
- Rework MIC Interpretation
- Use different cartridges
- Use different species
- Use different VITEK sources (UKM)
- Differentiate between oral and non oral?
- What shall happen with EUCAST values that are doublets (e.g.: due extra information)

## Restrictions
- Only E.coli
- Only AST-N428 cartridge
- Only UKM VITEK files