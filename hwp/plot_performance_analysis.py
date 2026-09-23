"""
plot_performance_analysis.py 

Author: Mattia Bianchessi
Email: mbianchessi@apc.in2p3.fr
Created: 2026-09-22
"""

import numpy as np
import matplotlib.pyplot as plt

from my_themes import set_theme, get_color, apply_theme
from plot_functions import save_pdf
set_theme('publication')

data = np.loadtxt('performance.txt', dtype=str, delimiter=',')

date = data[:, 0]
date = date.astype('datetime64')
success = data[:,1].astype(float)
failure = data[:,2].astype(float)

total = success + failure
mask = total > 37

plt.ion()
fig, ax = plt.subplots()
ax.plot(date[mask], failure[mask]/total[mask]*100, marker='*')
fig.autofmt_xdate()
ax.set_xlabel('Date')
ax.set_ylabel('Percentage of failures')
ax.set_title('HWP failures over time')

fix_dates = np.array(['2026-07-08', '2026-07-10', '2026-07-16', '2026-07-17', '2026-07-20', '2026-07-30', '2026-08-05', '2026-08-06'], dtype='datetime64[D]')

for fix in fix_dates:
    ax.axvline(fix, ls='--', c=get_color(1))

save_pdf('hwp_performance')
