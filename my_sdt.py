"""
my_sdt.py - Signal detection analysis of visual search (human data only)

Project:  NRSC510A final project, University of British Columbia
Author:   Zeynep Alici (NOVA Lab)
Question: Do errors in hard visual search reflect lower sensitivity (d'),
          a shift in response bias (criterion c), or both?
Data:     Zhang et al. (2026), trial-level human data
          (all_participant_data_with_general_info.csv; link in README.md)

What it does, in order:
  1. Loads the data, removes practice trials and implausible reaction times.
  2. Labels each trial as hit, miss, false alarm or correct rejection and
     counts them in each of the 126 cells (2 dimensions x 9 conditions x 7 set sizes).
  3. Computes d' and c for every cell (add_sdt function).
  4. Bootstraps over participants (1,000 repetitions, seed 510) for 95% intervals.
  5. Draws the d' and c figures, a bootstrap histogram, and the slopes per doubling of set size.
  6. Compares 3D with 2D, and compares slower with faster participants.

How to run: put the data CSV in the same folder as this file, then run
  python my_sdt.py
It takes a few minutes (the bootstrap is the slow part). Outputs (CSV files
and PNG figures) are written to the same folder.
"""
import pandas as pd 
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
# Larger fonts so the figures are readable on slides
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})
# ---------------------------------------------------------------
# Function: counts -> rates -> z-scores -> d' and criterion c
# ---------------------------------------------------------------
def add_sdt(table):
    """Take a table with hit, miss, false_alarm and correct_rejection columns
    and add hit_rate, fa_rate, d_prime and criterion columns."""
    table = table.copy()   # work on a copy so the original is not changed

    # Rates, with the log-linear correction (+0.5 and +1)
    table["hit_rate"] = (table["hit"] + 0.5) / (table["hit"] + table["miss"] + 1)
    table["fa_rate"] = (table["false_alarm"] + 0.5) / (table["false_alarm"] + table["correct_rejection"] + 1)

    # z-scores, then d' and criterion
    z_hit = norm.ppf(table["hit_rate"])
    z_fa = norm.ppf(table["fa_rate"])
    table["d_prime"] = z_hit - z_fa
    table["criterion"] = -0.5 * (z_hit + z_fa)

    return table
# ---------------------------------------------------------------
# 1. Load and clean the data
# ---------------------------------------------------------------
df = pd.read_csv("all_participant_data_with_general_info.csv")
print(len(df))
print(df.columns)
print(df.head())
# Remove practice trials
df = df[~df["category"].str.contains("practice", case=False)]
print(len(df))
# Keep only reaction times between 150 and 20,000 ms
df = df[(df["response_time"] >= 150) & (df["response_time"] <= 20000)]
print(len(df))
print(df.columns.tolist())
print(df["participant_index"].nunique())
# The category text holds the dimension (first character: 2 or 3) and the condition ("lv" + number)
df["dimension"] = df["category"].str[0].astype(int)
df["condition"] = df["category"].str.extract(r"lv(\d+)")[0].astype(int)
print(df[["category", "dimension", "condition"]].head())
print(df.groupby(["dimension", "condition"]).size())
# ---------------------------------------------------------------
# 2. Label every trial: hit, miss, false alarm or correct rejection
#    ("present"/"absent" = what the target was / what the participant answered)
# ---------------------------------------------------------------
df["hit"] = (df["target"] == "present") & (df["key_meaning"] == "present")
df["miss"] = (df["target"] == "present") & (df["key_meaning"] == "absent")
df["false_alarm"] = (df["target"] == "absent") & (df["key_meaning"] == "present")
df["correct_rejection"] = (df["target"] == "absent") & (df["key_meaning"] == "absent")
print(df[["hit", "miss", "false_alarm", "correct_rejection"]].sum())
# Count the four outcomes in each of the 126 cells
counts = (df.groupby(["dimension", "condition", "set_size"])
            [["hit", "miss", "false_alarm", "correct_rejection"]]
            .sum()
            .reset_index())
print(counts.head())
print(len(counts))
# ---------------------------------------------------------------
# 3. d' and c for every cell
# ---------------------------------------------------------------
counts = add_sdt(counts)
# ---------------------------------------------------------------
# 4. Bootstrap over participants (for 95% confidence intervals)
# ---------------------------------------------------------------
# Counts for each participant separately, in each cell
types = ["hit", "miss", "false_alarm", "correct_rejection"]

per_person = (df.groupby(["participant_index", "dimension", "condition", "set_size"])[types]
                .sum()
                .reset_index())

print(len(per_person))
print(per_person[types].sum())
# A random number generator with a fixed seed, so results are the same every run
rng = np.random.default_rng(510)

def one_bootstrap(per_person, rng):
    """Resample participants (within each dimension) and recompute d' and c."""
    pieces = []
    for dim in [2, 3]:
        sub = per_person[per_person["dimension"] == dim].copy()
        ids = sub["participant_index"].unique()

        # Pick as many participants as there were, WITH replacement
        chosen = rng.choice(ids, size=len(ids), replace=True)

        # How many times was each participant picked? (0 if never picked)
        times = pd.Series(chosen).value_counts()
        sub["weight"] = sub["participant_index"].map(times).fillna(0)

        # Picked twice = counted twice; never picked = counted zero times
        for t in types:
            sub[t] = sub[t] * sub["weight"]
        pieces.append(sub)

    boot = pd.concat(pieces)
    boot_counts = (boot.groupby(["dimension", "condition", "set_size"])[types]
                       .sum()
                       .reset_index())
    return add_sdt(boot_counts)

result = one_bootstrap(per_person, rng)
print(len(result))
print(counts.loc[48, "d_prime"], result.loc[48, "d_prime"])
n_boot = 1000  # number of bootstrap repetitions

# Empty tables to fill: one row per repetition, one column per cell (126)
boot_d = np.zeros((n_boot, len(counts)))
boot_c = np.zeros((n_boot, len(counts)))

for i in range(n_boot):
    res = one_bootstrap(per_person, rng)
    boot_d[i, :] = res["d_prime"].to_numpy()
    boot_c[i, :] = res["criterion"].to_numpy()
    if i % 50 == 0:
        print("repetition", i)

print(boot_d.shape)
print(boot_d[:, 48])
# 95% confidence interval for every cell: the 2.5th and 97.5th percentiles
counts["d_low"] = np.percentile(boot_d, 2.5, axis=0)
counts["d_high"] = np.percentile(boot_d, 97.5, axis=0)
counts["c_low"] = np.percentile(boot_c, 2.5, axis=0)
counts["c_high"] = np.percentile(boot_c, 97.5, axis=0)

print(counts.loc[48, ["d_prime", "d_low", "d_high", "criterion", "c_low", "c_high"]])

# Save the full table: counts, rates, d', c and their 95% intervals
counts.to_csv("my_counts_ci.csv", index=False)
print(counts[(counts.dimension == 2) & (counts.condition == 7) & (counts.set_size == 256)])
# ---------------------------------------------------------------
# 5. Figures
# ---------------------------------------------------------------
# Figure A: d' against set size, all 9 conditions (hard to read: 9 lines per panel)
# One panel for 2D, one for 3D
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, dim in zip(axes, [2, 3]):
    for cond in range(1, 10):
        g = counts[(counts.dimension == dim) & (counts.condition == cond)]
        line, = ax.plot(g["set_size"], g["d_prime"], marker="o", label=f"C{cond}")
        ax.fill_between(g["set_size"], g["d_low"], g["d_high"], color=line.get_color(), alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("Set size")
    ax.set_title(f"{dim}D conditions")

axes[0].set_ylabel("Sensitivity d'")
axes[1].legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("my_dprime_ci.png", dpi=200)

# Figure B: same as A but only 4 conditions (C1 = easiest feature search,
# C4 = medium feature search, C7 and C9 = conjunction search, hardest).
# I made this subset because the plot with all 9 conditions was too messy to read.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, dim in zip(axes, [2, 3]):
    for cond in [1, 4, 7, 9]:
        g = counts[(counts.dimension == dim) & (counts.condition == cond)]
        line, = ax.plot(g["set_size"], g["d_prime"], marker="o", label=f"C{cond}")
        ax.fill_between(g["set_size"], g["d_low"], g["d_high"], color=line.get_color(), alpha=0.15)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("Set size")
    ax.set_title(f"{dim}D conditions")

axes[0].set_ylabel("Sensitivity d'")
axes[1].legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("my_dprime_ci_subset.png", dpi=200)

# Figure C: criterion c against set size, all 9 conditions (no intervals)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

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

# Figure D: criterion c, all 9 conditions, with 95% intervals
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, dim in zip(axes, [2, 3]):
    for cond in range(1, 10):
        g = counts[(counts.dimension == dim) & (counts.condition == cond)]
        line, = ax.plot(g["set_size"], g["criterion"], marker="o", label=f"C{cond}")
        ax.fill_between(g["set_size"], g["c_low"], g["c_high"], color=line.get_color(), alpha=0.15)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)   # c = 0 means no bias
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("Set size")
    ax.set_title(f"{dim}D conditions")

axes[0].set_ylabel("Criterion c")
axes[1].legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("my_criterion_ci.png", dpi=200)

# Figure E: criterion c, subset of 4 conditions, with 95% intervals
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

for ax, dim in zip(axes, [2, 3]):
    for cond in [1, 4, 7, 9]:
        g = counts[(counts.dimension == dim) & (counts.condition == cond)]
        line, = ax.plot(g["set_size"], g["criterion"], marker="o", label=f"C{cond}")
        ax.fill_between(g["set_size"], g["c_low"], g["c_high"], color=line.get_color(), alpha=0.15)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)   # c = 0 means no bias
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32, 64, 128, 256])
    ax.set_xticklabels([4, 8, 16, 32, 64, 128, 256])
    ax.set_xlabel("Set size")
    ax.set_title(f"{dim}D conditions")

axes[0].set_ylabel("Criterion c")
axes[1].legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
fig.tight_layout()
fig.savefig("my_criterion_ci_subset.png", dpi=200)

# Figure F: histogram of the 1,000 bootstrap d' values for one cell (2D, C7, set size 256)
# The red dashed lines are the 95% interval; the black line is the real d'
plt.figure()
plt.hist(boot_d[:, 48], bins=30)
plt.axvline(counts.loc[48, "d_low"], color="red", linestyle="--")
plt.axvline(counts.loc[48, "d_high"], color="red", linestyle="--")
plt.axvline(counts.loc[48, "d_prime"], color="black")
plt.xlabel("Bootstrap d' for 2D, C7, set size 256")
plt.ylabel("Number of repetitions")
plt.savefig("my_bootstrap_hist.png", dpi=200)
# ============================================================
# Slopes of d' and c against log2(set size), with 95% intervals
# ============================================================

def slope_per_row(y, x):
    """Least-squares slope of y against x.
    y can be one list of 7 values, or a table with one row per
    bootstrap repetition and 7 columns (one per set size)."""
    x_centred = x - x.mean()
    y = np.atleast_2d(y)                                   # make sure y has rows
    y_centred = y - y.mean(axis=1, keepdims=True)          # subtract each row's mean
    return (y_centred * x_centred).sum(axis=1) / (x_centred ** 2).sum()


slope_rows = []

for (dim, cond), g in counts.groupby(["dimension", "condition"]):
    idx = g.index.to_numpy()                       # which of the 126 rows belong to this group
    x = np.log2(g["set_size"].to_numpy())          # 4, 8, ..., 256 becomes 2, 3, ..., 8

    # Slope from the real data
    d_slope = slope_per_row(g["d_prime"].to_numpy(), x)[0]
    c_slope = slope_per_row(g["criterion"].to_numpy(), x)[0]

    # Slope from every bootstrap repetition (the same 1000 resamples as before)
    d_boot = slope_per_row(boot_d[:, idx], x)
    c_boot = slope_per_row(boot_c[:, idx], x)

    slope_rows.append({
        "dimension": dim, "condition": cond,
        "d_slope": d_slope,
        "d_slope_low": np.percentile(d_boot, 2.5),
        "d_slope_high": np.percentile(d_boot, 97.5),
        "c_slope": c_slope,
        "c_slope_low": np.percentile(c_boot, 2.5),
        "c_slope_high": np.percentile(c_boot, 97.5),
    })

slopes = pd.DataFrame(slope_rows)
print(slopes.round(2))
slopes.to_csv("my_slopes.csv", index=False)


# ---- Figures G and H: slope for each condition, with its interval ----
for measure, label, fname in [("d", "d'", "my_dprime_slopes.png"),
                              ("c", "criterion c", "my_criterion_slopes.png")]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, dim in zip(axes, [2, 3]):
        s = slopes[slopes["dimension"] == dim]
        err_low = s[f"{measure}_slope"] - s[f"{measure}_slope_low"]
        err_high = s[f"{measure}_slope_high"] - s[f"{measure}_slope"]

        ax.errorbar(s["condition"], s[f"{measure}_slope"],
                    yerr=[err_low, err_high], fmt="o", capsize=4)
        ax.axhline(0, color="black", linestyle="--")   # zero = no change with set size
        ax.set_xlabel("Condition")
        ax.set_title(f"{dim}D conditions")

    axes[0].set_ylabel(f"Slope of {label} per doubling of set size")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)

# ===============================================================
# 6a. 2D vs 3D: difference in d' and c for each condition
#     (the 2D and 3D groups are different people, so their bootstrap
#     results can be subtracted directly)
# ===============================================================
# counts is sorted by dimension, so rows 0-62 are 2D and rows 63-125 are 3D
diff_d = boot_d[:, 63:] - boot_d[:, :63]      # 3D minus 2D, shape (1000, 63)
diff_c = boot_c[:, 63:] - boot_c[:, :63]

two = counts[counts["dimension"] == 2].reset_index(drop=True)
three = counts[counts["dimension"] == 3].reset_index(drop=True)

# Average the difference over the 7 set sizes, for each of the 9 conditions
obs_d = (three["d_prime"] - two["d_prime"]).to_numpy().reshape(9, 7).mean(axis=1)
obs_c = (three["criterion"] - two["criterion"]).to_numpy().reshape(9, 7).mean(axis=1)
boot_d_cond = diff_d.reshape(n_boot, 9, 7).mean(axis=2)     # (1000, 9)
boot_c_cond = diff_c.reshape(n_boot, 9, 7).mean(axis=2)

dim_table = pd.DataFrame({
    "condition": range(1, 10),
    "d_diff": obs_d,
    "d_low": np.percentile(boot_d_cond, 2.5, axis=0),
    "d_high": np.percentile(boot_d_cond, 97.5, axis=0),
    "c_diff": obs_c,
    "c_low": np.percentile(boot_c_cond, 2.5, axis=0),
    "c_high": np.percentile(boot_c_cond, 97.5, axis=0),
})
print(dim_table.round(2))
dim_table.to_csv("my_2d_vs_3d.csv", index=False)

# Why dots with error bars: each condition gives ONE difference with a 95% interval,
# and the dashed line at 0 shows at a glance whether the interval crosses 'no difference'.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name, col in zip(axes, ["d", "c"], ["Sensitivity d'", "Criterion c"]):
    y = dim_table[name + "_diff"]
    err = [y - dim_table[name + "_low"], dim_table[name + "_high"] - y]
    ax.errorbar(dim_table["condition"], y, yerr=err, fmt="o", capsize=5)
    ax.axhline(0, color="grey", linestyle="--")
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f"C{i}" for i in range(1, 10)])
    ax.set_xlabel("Condition")
    ax.set_ylabel(f"3D minus 2D ({col})")
    ax.set_title(col)
fig.tight_layout()
fig.savefig("my_2d_vs_3d.png", dpi=200)

# ===============================================================
# 6b. Faster vs slower participants (median split on each person's median RT,
#     within each dimension). This compares different people, not fast and
#     slow trials of the same person.
# ===============================================================
med_rt = df.groupby("participant_index")["response_time"].median().rename("med_rt").to_frame()
med_rt["dimension"] = per_person.groupby("participant_index")["dimension"].first()
group_median = med_rt.groupby("dimension")["med_rt"].transform("median")
med_rt["speed"] = np.where(med_rt["med_rt"] > group_median, "slow", "fast")
print(med_rt.groupby(["dimension", "speed"]).size())

per_person = per_person.join(med_rt["speed"], on="participant_index")

def speed_counts(table):
    out = (table.groupby(["speed", "dimension", "condition", "set_size"])[types]
                .sum().reset_index())
    return add_sdt(out)      # sorted: 126 'fast' rows, then 126 'slow' rows

def slow_minus_fast(table, col):
    """slow minus fast, averaged over set sizes -> 18 values (2 dims x 9 conditions)"""
    v = table[col].to_numpy()
    return (v[126:] - v[:126]).reshape(18, 7).mean(axis=1)

def one_speed_bootstrap(rng):
    pieces = []
    for (dim, sp), sub in per_person.groupby(["dimension", "speed"]):
        sub = sub.copy()
        ids = sub["participant_index"].unique()
        chosen = rng.choice(ids, size=len(ids), replace=True)
        times = pd.Series(chosen).value_counts()
        sub["weight"] = sub["participant_index"].map(times).fillna(0)
        for t in types:
            sub[t] = sub[t] * sub["weight"]
        pieces.append(sub)
    return speed_counts(pd.concat(pieces))

obs = speed_counts(per_person)
obs_c_diff = slow_minus_fast(obs, "criterion")
obs_d_diff = slow_minus_fast(obs, "d_prime")

bc = np.zeros((n_boot, 18))
bd = np.zeros((n_boot, 18))
for i in range(n_boot):
    r = one_speed_bootstrap(rng)
    bc[i] = slow_minus_fast(r, "criterion")
    bd[i] = slow_minus_fast(r, "d_prime")

rt_table = pd.DataFrame({
    "dimension": [2] * 9 + [3] * 9,
    "condition": list(range(1, 10)) * 2,
    "c_diff": obs_c_diff, "c_low": np.percentile(bc, 2.5, axis=0), "c_high": np.percentile(bc, 97.5, axis=0),
    "d_diff": obs_d_diff, "d_low": np.percentile(bd, 2.5, axis=0), "d_high": np.percentile(bd, 97.5, axis=0),
})
print(rt_table.round(2))
rt_table.to_csv("my_c_vs_rt.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
for ax, name, lab in zip(axes, ["c", "d"], ["criterion c", "sensitivity d'"]):
    for dim, shift, color in [(2, -0.1, "tab:blue"), (3, 0.1, "tab:orange")]:
        g = rt_table[rt_table["dimension"] == dim]
        y = g[name + "_diff"].to_numpy()
        err = [y - g[name + "_low"].to_numpy(), g[name + "_high"].to_numpy() - y]
        ax.errorbar(g["condition"] + shift, y, yerr=err, fmt="o", capsize=4, color=color, label=f"{dim}D")
    ax.axhline(0, color="grey", linestyle="--")
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f"C{i}" for i in range(1, 10)])
    ax.set_xlabel("Condition")
    ax.set_ylabel(f"Slower minus faster half ({lab})")
axes[0].legend()
fig.tight_layout()
fig.savefig("my_c_vs_rt.png", dpi=200)
plt.show()