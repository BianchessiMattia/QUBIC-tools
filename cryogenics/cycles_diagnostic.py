import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from pathlib import Path

from qubic_sys.hk_utilities import download_groups, read_hk_file, HK_QUANTITIES, plot_temperature, load_hk, load_fridge_cycles

from my_themes import set_theme, get_color, apply_theme
from plot_functions import plot_flags, on_click
set_theme('terminal')

CYCLES_DIR = Path('/home/mattia/se/cryogenics/')


start_date = np.datetime64('2026-10-01 00:00:00')
#end_date = np.datetime64('2026-09-30 23:59:00')
end_date = np.datetime64('now')


# -------------------------------------------------------------------
# 1 K
# ------------------------------------------------------------------
download_groups(['1K'])

data = {}
for name, filename in HK_QUANTITIES['1K'].items():
    data[name] = load_hk(filename, start_date, end_date)

fig,ax = plot_temperature(data, '1K fridge cycles')
plt.connect('button_press_event', on_click)

#plt.savefig(f'figures/1K_fridge_cycles_{end_date}.png', dpi=300)
#ax.axvline(np.datetime64('2026-10-08 18:10:00'), ls='--', color=get_color(6))
#plot_flags(ax, {np.datetime64('2026-09-23 17:00:00'): 'compressors off'})

cycles_1K_starts, cycles_1K_ends = load_fridge_cycles(CYCLES_DIR / 'cycles_1Kstage.csv', start_date, end_date)

ax.axhline(1.2, ls='--', lw=2, color=get_color(7))
for start in cycles_1K_starts:
    ax.axvline(start, ls='--', lw=2, color=get_color(6))
for end in cycles_1K_ends:
    ax.axvline(end, ls='--', lw=2, color=get_color(6))


# -------------------------------------------------------------------
# 300 mK
# ------------------------------------------------------------------
download_groups(['300mK'])

data = {}
for name, filename in HK_QUANTITIES['300mK'].items():
    data[name] = load_hk(filename, start_date, end_date)

fig,ax = plot_temperature(data, '300mK fridge cycles')
plt.connect('button_press_event', on_click)

#plt.savefig(f'figures/1K_fridge_cycles_{end_date}.png', dpi=300)
#ax.axvline(np.datetime64('2026-10-08 18:10:00'), ls='--', color=get_color(6))
#plot_flags(ax, {np.datetime64('2026-09-23 17:00:00'): 'compressors off'})

cycles_300mK_starts, cycles_300mK_ends = load_fridge_cycles(CYCLES_DIR / 'cycles_300mKstage.csv', start_date, end_date)

ax.axhline(0.340, ls='--', lw=2, color=get_color(7))
for start in cycles_300mK_starts:
    ax.axvline(start, ls='--', lw=2, color=get_color(6))
for end in cycles_300mK_ends:
    ax.axvline(end, ls='--', lw=2, color=get_color(6))
