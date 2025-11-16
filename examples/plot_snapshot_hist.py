import numpy as np
import matplotlib.pyplot as plt
from quokka2s.tables import load_table, plot_sampling_histogram



table = load_table("path/to/your_table.npz")
ax = plot_sampling_histogram(table, samples, log_space=True)
ax.set_title("Snapshot sampling vs DESPOTIC failures")
plt.savefig("snapshot_hist.png", dpi=200)