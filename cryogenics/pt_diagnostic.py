import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from qubic_sys.hk_utilities import download_groups, read_hk_file, HK_QUANTITIES, plot_temperature, load_hk

from my_themes import set_theme, get_color, apply_theme
from plot_functions import plot_flags, on_click
set_theme('terminal')

download_groups(['pt'])

start_date = np.datetime64('2026-10-01 00:00:00')
#end_date = np.datetime64('2026-10-06 10:00:00')
end_date = np.datetime64('now')

selected_quantities = ['PT2_S2_CH', 'PT2_S2', 'fridge_plate']
data = {}
for quantity in selected_quantities:
    data[quantity] = load_hk(HK_QUANTITIES['pt'][quantity], start_date, end_date)

plot_temperature(data, 'Pulse tube temperature diagnostic')
plt.connect('button_press_event', on_click)

#plt.savefig(f'figures/pt_{end_date}.png', dpi=300)
#ax.axvline(np.datetime64('2026-09-23 13:50:00'), ls='--', color=get_color(6)) 
