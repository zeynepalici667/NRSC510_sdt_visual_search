"""
my_bellcurve.py - draws the signal detection model figure used in the introduction.
The two numbers below (d' = 1.79, c = 0.67) are the real values for one cell of the data:
2D, condition C7, set size 256 (see my_counts_ci.csv). Run: python my_bellcurve.py
Author: Zeynep Alici (NRSC510A final project)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# The two numbers that describe the picture
d_prime = 1.79   # distance between the two peaks
c = 0.67         # where the criterion line sits relative to the midpoint

# x axis: 500 evenly spaced points from -4 to 6
x = np.linspace(-4, 6, 500)

# Height of each bell curve at every x
absent = norm.pdf(x, loc=0, scale=1)         # target absent: centred at 0
present = norm.pdf(x, loc=d_prime, scale=1)  # target present: centred at d'

print(len(x))
print(absent.max())
# Draw the two curves
plt.plot(x, absent, label="Target absent")
plt.plot(x, present, label="Target present")

plt.xlabel("Evidence for 'target present'")
plt.ylabel("Probability density")
# Criterion: the point where the person switches from "absent" to "present"
midpoint = d_prime / 2          # halfway between the two peaks (no bias)
criterion_line = midpoint + c   # shift the line by c

plt.axvline(criterion_line, color="purple", linestyle="--", label="Criterion")
# Pick out the x positions to the right of the criterion line
right = x >= criterion_line

# Shade the area under each curve to the right of the line
plt.fill_between(x[right], present[right], color="green", alpha=0.3, label="Hits")
plt.fill_between(x[right], absent[right], color="red", alpha=0.3, label="False alarms")

# Arrow showing d' (distance between the two peaks)
plt.annotate("", xy=(d_prime, 0.43), xytext=(0, 0.43),
             arrowprops=dict(arrowstyle="<->"))
plt.text(d_prime / 2, 0.44, "d' = 1.79", ha="center")

plt.ylim(0, 0.5)   # leave room at the top for the arrow
plt.title("Signal detection model for one cell (2D, C7, set size 256)")
plt.legend()
print("hit rate:", 1 - norm.cdf(criterion_line, loc=d_prime))
print("false-alarm rate:", 1 - norm.cdf(criterion_line, loc=0))
plt.savefig("my_bellcurve.png", dpi=200)
plt.show()
