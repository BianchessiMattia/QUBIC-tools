"""
plot_rotation_time.py

Author: Mattia Bianchessi
Email: mbianchessi@apc.in2p3.fr
Created: 2026-09-22
"""

import numpy as np
import matplotlib.pyplot as plt
from my_themes import set_theme, get_color, apply_theme
from plot_functions import save_pdf
set_theme('publication')

data = np.loadtxt('rotation_time.txt', dtype=str, delimiter=',')
date = data[:, 0]
date = date.astype('datetime64')
success_median = data[:,3].astype(float)
success_mean = data[:, 2].astype(float)
success_std = data[:,4].astype(float)

#plt.ion()
fig, ax = plt.subplots()
fig.autofmt_xdate()
ax.set_xlabel('Date')
ax.set_ylabel('HWP rotation time [s]')

ax.errorbar(date, success_mean, success_std, c=get_color(0), label='Mean time + std errorbar')
ax.scatter(date, success_median, marker='*', c=get_color(1), label='Median time')
plt.legend()

save_pdf('hwp_rotation_time')
