"""Poster versions of the main figures: bigger text, fewer elements.

    python poster_figures.py       (after run_pipeline.py)
    python case_studies.py --poster

Writes figures/poster/.
"""
import json
import os

import matplotlib.pyplot as plt

from src import figures as fg

POSTER_RC = {"font.size": 16, "axes.titlesize": 20, "axes.labelsize": 16, "xtick.labelsize": 15,
             "ytick.labelsize": 16, "legend.fontsize": 15}
os.makedirs("figures/poster", exist_ok=True)
R = json.load(open("results/metrics.json"))

with plt.rc_context(POSTER_RC):
    fg.importance_comparison(R["agreement"]["importance_shap"], R["agreement"]["importance_ebm"],
                             "XGBoost + SHAP", "EBM (glass box)", "What matters most, per method",
                             "figures/poster/importance.png", top_n=6, width=9.5, row_h=0.38)
    fg.iterations_plot(R["iterations"], "figures/poster/iterations.png",
                       title="Each fix, tested on the unseen 2025 season", figsize=(9.5, 3.5),
                       labels={"v1 original (absolute temp, no SC-upcoming)": "v1 original",
                               "v2 + SC/VSC deployed feature": "v2 + safety-car feature",
                               "No temperature (control)": "No temperature (control)",
                               "v3 final (temp change instead of absolute)": "v3 final (temp change)"})
print("wrote figures/poster/")
