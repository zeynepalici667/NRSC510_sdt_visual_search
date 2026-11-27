# Perceptual Sensitivity and Response Bias in Visual Search: A Signal Detection Analysis

NRSC510A final project, University of British Columbia. Author: Zeynep Alici (NOVA Lab).

- **Dataverse** (data outputs, proposal, slides, license and justification): https://demo.borealisdata.ca/previewurl.xhtml?token=5c3b5858-18f5-4228-807f-08e9bf6ee9dd
- **Data analysed:** Zhang et al. (2026), human participants only: https://doi.org/10.4121/e2b3e80f-8983-478b-a93f-8c8858fe8f96

## Research question
When people make errors in hard visual search, do those errors reflect lower sensitivity (d'), a shift in response bias (criterion c), or both?

## What the code does (coding methodology)
All steps are in `my_sdt.py`, in this order:
1. **Load and clean.** Removes practice trials and reaction times outside 150-20,000 ms (168,487 -> 157,259 -> 157,042 trials; 1,250 participants).
2. **Label and count.** Each trial is a hit, miss, false alarm or correct rejection. These are counted in each of 126 cells (2 dimensions x 9 conditions x 7 set sizes, 4-256).
3. **d' and c per cell** (function `add_sdt`). Hit and false-alarm rates use the log-linear correction (add 0.5 to the counts and 1 to the totals; Hautus, 1995). Rates become z-scores with `scipy.stats.norm.ppf`, then d' = z(hit) - z(false alarm) and c = -0.5 x (z(hit) + z(false alarm)). A positive c means a lean toward answering "absent".
4. **Bootstrap over participants.** Participants are resampled with replacement within each dimension, 1,000 repetitions, random seed 510. The 95% interval is the 2.5th to 97.5th percentile. 1,000 repetitions is what my laptop could run in reasonable time.
5. **Figures and slopes.** d' and c against set size (all conditions, and a subset of C1, C4, C7, C9 because the all-conditions plot was too crowded), a bootstrap histogram, and the slope of d' and c per doubling of set size (least squares against log2 set size) with bootstrap intervals.
6. **Extra analyses.** 3D minus 2D differences per condition, and slower minus faster half of participants (median split of each person's median reaction time, within each dimension). The second compares different people, not fast and slow trials of one person.

`my_bellcurve.py` draws the signal detection model figure (d' = 1.79, c = 0.67, the values for 2D, C7, set size 256).

## Files and naming
| Path | What it is |
|---|---|
| `my_sdt.py` | The whole analysis. Run this. |
| `my_bellcurve.py` | Signal detection model figure |
| `requirements.txt` | Packages with versions |
| `expected_output/` | Results from my run, to compare with yours (CSV files and PNG figures) |
| `LICENSE.txt` | License (CC BY 4.0) |

Naming: files I made start with `my_`; output files are named after what they contain (`my_counts_ci.csv` = counts with confidence intervals; `_subset` = only conditions C1, C4, C7, C9; `_slopes` = slope per doubling of set size).

Output files written when you run `my_sdt.py`: `my_counts_ci.csv` (126 rows: counts, rates, d', c and 95% intervals), `my_slopes.csv` (18 rows), `my_2d_vs_3d.csv` (9 rows), `my_c_vs_rt.csv` (18 rows), and the figures `my_dprime_ci.png`, `my_dprime_ci_subset.png`, `my_criterion.png`, `my_criterion_ci.png`, `my_criterion_ci_subset.png`, `my_bootstrap_hist.png`, `my_dprime_slopes.png`, `my_criterion_slopes.png`, `my_2d_vs_3d.png`, `my_c_vs_rt.png`.

## Requirements
Versions used for the results in `expected_output/`:
- Python 3.14.7 (https://www.python.org)
- NumPy 2.5.4 (https://numpy.org)
- pandas 3.1.0 (https://pandas.pydata.org)
- SciPy 1.18.1 (https://scipy.org)
- Matplotlib 3.11.2 (https://matplotlib.org)

## How to reproduce (step by step)
1. Download this repository (green "Code" button, then "Download ZIP") and unzip it.
2. Download `all_participant_data_with_general_info.csv` from https://doi.org/10.4121/e2b3e80f-8983-478b-a93f-8c8858fe8f96 and put it in the same folder as `my_sdt.py`.
3. Open a terminal in that folder. Optional but recommended: `python3 -m venv .venv`, then `source .venv/bin/activate` (Mac/Linux) or `.venv\Scripts\activate` (Windows).
4. Install the packages: `pip install -r requirements.txt`
5. Run: `python my_sdt.py`. It takes a few minutes (the bootstrap is the slow part) and prints progress every 50 repetitions.
6. The CSV files and figures appear in the same folder, and the figures open in windows at the end.
7. Optional: `python my_bellcurve.py` redraws the signal detection model figure.

## Check your results (known outputs)
The random seed is fixed (seed 510), so you should get the same numbers (rounded):

| Cell | d' | 95% CI | c | 95% CI |
|---|---|---|---|---|
| 2D, C1, set size 256 | 4.53 | 4.20-5.18 | 0.27 | 0.10-0.57 |
| 2D, C7, set size 256 | 1.79 | 1.60-2.00 | 0.67 | 0.58-0.76 |
| 2D, C9, set size 256 | 1.00 | 0.85-1.13 | 0.32 | 0.25-0.40 |

The script also prints the trial counts 168,487 -> 157,259 -> 157,042 and 1250 participants. Your `my_counts_ci.csv` should match `expected_output/my_counts_ci.csv`.

## Main results
- d' falls with set size in the hard conditions (slope below 0 in 12 of 18 dimension x condition groups, -0.14 to -0.44 per doubling), and is flat in C1, C2 and C5.
- c rises with set size (slope above 0 in 13 of 18 groups, +0.03 to +0.12 per doubling): people lean more toward "absent" as displays get larger.
- Errors in hard search therefore reflect both lower sensitivity and a more conservative criterion. 2D and 3D behave alike; slower participants have higher d' but not a different c.

## License
CC BY 4.0 (same as the Dataverse). See `LICENSE.txt`. Please also cite Zhang et al. (2026), who own the data.

## References
- Hautus, M. J. (1995). Corrections for extreme proportions and their biasing effects on estimated values of d'. Behavior Research Methods, Instruments, & Computers, 27(1), 46-51. https://doi.org/10.3758/BF03203619
- Stanislaw, H., & Todorov, N. (1999). Calculation of signal detection theory measures. Behavior Research Methods, Instruments, & Computers, 31(1), 137-149. https://doi.org/10.3758/BF03207704
- Zhang, R., de Winter, J. C. F., Dodou, D., Seyffert, H., & Eisma, Y. B. (2026). Human-like attention? A psychophysical comparison of visual search in humans and MLLMs. Computational Brain & Behavior. https://doi.org/10.1007/s42113-026-00333-4

## AI assistance
Code was written with guidance from Claude (Anthropic).
