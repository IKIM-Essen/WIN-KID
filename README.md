# WIN-KID

## ToDo
- Differentiate between oral and non oral?
- Use different cartridges
- Use different species
- What shall happen with EUCAST values that are doublets (e.g.: due extra information))

## Interpreter
- Antibiotics from a vitek file that can not be matched to an EUCAST antibiotic are skipped and logged via print
- For a match, the name of the VITEK antibiotic must be found in full in the EUCAST table
- If several values are found, the one with the highest similarity score is used. If two values have the same similarity, any one is used.
- If the selected value falls below a certain similarity score, it is also discarded and logged via print.
- All VITEK values with "<" are automatically "S"
- The ">" "=" are removed from the VITEK files and directly compared with the EUCAST values