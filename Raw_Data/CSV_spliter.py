import pandas as pd

df = pd.read_csv("data_Sigmoid.csv")

nodes = df["Model_Architecture"].unique()

for node in nodes:
    activation_df = df[df["Model_Architecture"] == node]
    activation_df.to_csv(f"data_Sigmoid_{node}.csv", index=False)
