import pandas as pd

df = pd.DataFrame(pd.read_csv("../binary.csv"))

df['pass'] = df['admit'] == 1
df = df.iloc[:, 1:]
df = df.rename(columns={'gre': "x", 'gpa': "y", "rank": "z"})
df.to_csv('binary_cross.csv')

print(df)


def normalize_column(column):
    return (column - column.min()) / (column.max() - column.min())


normalized_df = df.iloc[:, 0:3].apply(normalize_column)
normalized_df['pass'] = df['pass']

print(normalized_df)
normalized_df.to_csv('binary_cross_normalized.csv')
