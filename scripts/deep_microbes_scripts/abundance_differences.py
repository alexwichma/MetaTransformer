import pandas as pd

input_path = "/share/ebuschon/data/dm_original_mock_abundances/averaged.csv"

df = pd.read_csv(input_path)

df = df[df["Prediction(Ours)"] > 0]
difference = (df["Prediction(Ours)"] - df["Prediction(Original)"]).abs()
percentage = difference / df["Prediction(Ours)"]

print(percentage.min())
print(percentage.max())
print(percentage.mean())