import pandas as pd

df = pd.read_pickle("TWEET-FID/LREC_BSC/train.p")
print(df.head())
print(df.info())