import pandas as pd

df = pd.read_csv('/local/work/julian/WIN-KID-local/development/WIN-KID/resources/UKE_2024/Bacteria_2024.csv')
replacement = pd.read_csv('/local/work/julian/WIN-KID-local/development/WIN-KID/resources/UKE_2024/name_replacement.csv')
drug_class_df = pd.read_csv('/local/work/julian/WIN-KID-local/development/WIN-KID/resources/UKE_2024/drug_class.csv')

replacement_mapping = dict(zip(replacement['old'], replacement['new']))
df['Antibiotika'] = df['Antibiotika'].replace(replacement_mapping)
# Remove chars after space 
df['Antibiotika'] = df['Antibiotika'].astype(str).str.strip().str.split(' ').str[0]
# Remove chars after underscore
df['Antibiotika'] = (df['Antibiotika'].astype(str).str.strip().str.replace(r' .*$|_.*$', '', regex=True)
)
# Remove chars after '('
df['Antibiotika'] = (
    df['Antibiotika']
    .astype(str)
    .str.strip()
    .str.replace(r' .*$|_.*$|\(.*$', '', regex=True)
)
df['Antibiotika'] = df['Antibiotika'].astype(str).str.replace('/', '+')

print(len(df['Antibiotika'].unique()))
print(sorted(df['Antibiotika'].dropna().unique()))

# Drug class mapping
drug_class_mapping = dict(zip(drug_class_df['antibiotic'], drug_class_df['drug_class']))
df['drug_class'] = df['Antibiotika'].map(drug_class_mapping)

print(df['drug_class'].isna().sum())
print(df['Antibiotika'].value_counts())
print(df)

df.to_csv('/local/work/julian/WIN-KID-local/development/WIN-KID/resources/UKE_2024/Bacteria_2024_drug_class_V0.csv')