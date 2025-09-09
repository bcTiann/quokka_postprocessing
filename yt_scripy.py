import numpy as np
from matplotlib.colors import LogNorm
import yt
from yt.visualization.api import get_multi_plot


ds = yt.load("plt00500")

orient = "horizontal"
res = (800, 800)
fig, axes, colorbars = get_multi_plot(2, 3, colorbar=orient, bw=6)

plots = []
for ax in range(3):
    