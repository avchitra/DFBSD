import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('outbreaks.csv').dropna(subset=['Food']).groupby("Food").agg(
    {"Illnesses": "sum", "Hospitalizations": "sum", "Fatalities": "sum", "Food": "count"}
).rename(columns={"Food": "Count"}).reset_index()

df = df[['Food'].notna()] # preprocess

df["Severity"] = df["Illnesses"] + df["Hospitalizations"] * 5 + df["Fatalities"] * 20

df_sorted_by_food = df.sort_values(by="Count", ascending=False).head(10)[df["Food"] != "Multiple Foods"]

# plot

plt.figure(figsize=(12, 6))
sns.barplot(
    data=df_sorted_by_food, 
    x="Food", 
    y="Count", 
    hue="Severity", 
    palette="Reds", 
    dodge=False,
    width=0.8
)
plt.xticks(rotation=45, ha="right")
plt.xlabel("Food")
plt.ylabel("Frequency")
plt.title("Frequency of Food Appearances with Severity Indication")
plt.legend(title="Severity Score")

plt.show()