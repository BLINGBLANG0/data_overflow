import pandas as pd
df = pd.read_csv("train.csv")
df.info()                                    # dtypes, nulls
df.describe()                                # distributions
df["Purchased_Coverage_Bundle"].value_counts()  # class balance
df.isnull().sum()                       # missing values