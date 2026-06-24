import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

excel_path = 'data/Traning_Dataset.xlsx'
df = pd.read_excel(excel_path, engine='openpyxl')

df['label_comb'] = df[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']].apply(lambda row: ''.join(row.values.astype(str)), axis=1)

label_counts = Counter(df['label_comb'])

df['label_comb_adjusted'] = df['label_comb'].apply(lambda label: '00000001' if label_counts[label] < 5 else label)

train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['label_comb_adjusted'], random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_comb_adjusted'], random_state=42)

train_df.drop(['label_comb', 'label_comb_adjusted'], axis=1, inplace=True)
val_df.drop(['label_comb', 'label_comb_adjusted'], axis=1, inplace=True)
test_df.drop(['label_comb', 'label_comb_adjusted'], axis=1, inplace=True)

print("Training set size:", len(train_df))
print("Validation set size:", len(val_df))
print("Test set size:", len(test_df))

train_df.to_excel('train_set.xlsx', index=False)
val_df.to_excel('val_set.xlsx', index=False)
test_df.to_excel('test_set.xlsx', index=False)