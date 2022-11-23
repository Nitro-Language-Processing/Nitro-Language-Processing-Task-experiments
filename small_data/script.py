import pandas as pd

df = pd.read_csv("small_data/solution.csv")

df = df[["ner_label"]]
# df["Weight"] = 1 / 52
df.to_csv("small_data/good_solution.csv")