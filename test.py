import pandas as pd
df = pd.read_csv('eval/eval_dataset.csv')
print('Total questions:', len(df))
print('Columns:', df.columns.tolist())
print()
print('First 3 questions:')
for i, row in df.head(3).iterrows():
    print(f'Q: {row["question"]}')
    print(f'A: {row["ground_truth"]}')
    print()