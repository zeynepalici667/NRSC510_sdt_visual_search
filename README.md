# Perceptual sensitivity and response bias in visual search

Signal detection analysis (d′ and criterion c) of the human data from
Zhang et al. (2026). Course project for NRSC510A.

## Question
Do the differences between target-present and target-absent error rates in
hard visual search reflect lower sensitivity, a shift in response bias, or both?

## Files
- my_sdt.py: cleans the data, labels each trial (hit, miss, false alarm,
  correct rejection), and calculates d′ and c for each condition and set size
- my_counts.csv: counts, rates, d′ and c for each of the 126 cells
- my_dprime.png, my_criterion.png: d′ and c against set size

## How to run
1. Download the data (link below) and put the CSV in the same folder as my_sdt.py
2. Install the packages: pip install numpy pandas scipy matplotlib
3. Run: python my_sdt.py

## Data
Zhang et al. (2026): https://doi.org/10.4121/e2b3e80f-8983-478b-a93f-8c8858fe8f96.

## Status
Preliminary results. Confidence intervals and formal tests are still to come.

## References
- Hautus, M. J. (1995). Corrections for extreme proportions and their biasing
  effects on estimated values of d′. Behavior Research Methods, Instruments,
  & Computers, 27(1), 46-51. https://doi.org/10.3758/BF03203619
- Stanislaw, H., & Todorov, N. (1999). Calculation of signal detection theory
  measures. Behavior Research Methods, Instruments, & Computers, 31(1),
  137-149. https://doi.org/10.3758/BF03207704
- Zhang, R., de Winter, J. C. F., Dodou, D., Seyffert, H., & Eisma, Y. B. (2026).
  Human-like attention? A psychophysical comparison of visual search in humans
  and MLLMs. Computational Brain & Behavior.
  https://doi.org/10.1007/s42113-026-00333-4
  
