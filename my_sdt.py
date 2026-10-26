import pandas as pd 
from scipy.stats import norm
import matplotlib.pyplot as plt
print ("pandas works")
df = pd.read_csv("all_participant_data_with_general_info.csv")
print(len(df))
print(df.columns)
print(df.head())
df = df[~df["category"].str.contains("practice", case=False)]
print(len(df))
df = df[(df["response_time"] >= 150) & (df["response_time"] <= 20000)]
print(len(df))
df["dimension"] = df["category"].str[0].astype(int)
df["condition"] = df["category"].str.extract(r"lv(\d+)")[0].astype(int)
print(df[["category", "dimension", "condition"]].head())
print(df.groupby(["dimension", "condition"]).size())
df["hit"] = (df["target"] == "present") & (df["key_meaning"] == "present")
df["miss"] = (df["target"] == "present") & (df["key_meaning"] == "absent")
df["false_alarm"] = (df["target"] == "absent") & (df["key_meaning"] == "present")
df["correct_rejection"] = (df["target"] == "absent") & (df["key_meaning"] == "absent")
print(df[["hit", "miss", "false_alarm", "correct_rejection"]].sum())
counts = (df.groupby(["dimension", "condition", "set_size"])
            [["hit", "miss", "false_alarm", "correct_rejection"]]
            .sum()
            .reset_index())
print(counts.head())
print(len(counts))
counts.to_csv("my_counts.csv", index=False)
counts["hit_rate"] = (counts["hit"] + 0.5) / (counts["hit"] + counts["miss"] + 1)
counts["fa_rate"] = (counts["false_alarm"] + 0.5) / (counts["false_alarm"] + counts["correct_rejection"] + 1)
z_hit = norm.ppf(counts["hit_rate"])
z_fa = norm.ppf(counts["fa_rate"])
counts["d_prime"] = z_hit - z_fa
counts["criterion"] = -0.5 * (z_hit + z_fa)
print(counts[(counts.dimension == 2) & (counts.condition == 7) & (counts.set_size == 256)])
counts.to_csv("my_counts.csv", index=False)
# One panel for 2D, one for 3D
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, dim in zip(axes, [2, 3]):
    for cond in range(1, 10):
        g = counts[(counts.dimension == dim) & (counts.condition == cond)]
        ax.plot(g["set_size"], g["d_prime"], marker="o", label=f"C{cond}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("Set size")
    ax.set_title(f"{dim}D conditions")

axes[0].set_ylabel("Sensitivity d'")
axes[1].legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("my_dprime.png", dpi=200)
# Same layout as the d' figure, but for criterion c
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for ax, dim in zip(axes, [2, 3]):
    for cond in range(1, 10):
        g = counts[(counts.dimension == dim) & (counts.condition == cond)]
        ax.plot(g["set_size"], g["criterion"], marker="o", label=f"C{cond}")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)   # c = 0 means no bias
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("Set size")
    ax.set_title(f"{dim}D conditions")

axes[0].set_ylabel("Criterion c")
axes[1].legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("my_criterion.png", dpi=200)