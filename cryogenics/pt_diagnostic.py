import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from qubic_sys.hk_utilities import download_groups, read_hk_file, HK_QUANTITIES, plot_temperature

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
    print('Loading', quantity)
    dat = read_hk_file(HK_QUANTITIES['pt'][quantity])

    rangemask = (dat[0] >= start_date.astype(float)) & (dat[0] <= end_date.astype(float))
    temperature = dat[1][rangemask]
    time = dat[0][rangemask]
    date = (time * 1e3).astype('datetime64[ms]')

    data[quantity] = {
        'temperature': temperature,
        'time': time,
        'date': date,
    }

plot_temperature(data, 'Pulse tube temperature diagnostic')

plt.savefig(f'figures/pt_{end_date}.png', dpi=300)

plt.connect('button_press_event', on_click)

ax.axvline(np.datetime64('2026-09-23 13:50:00'), ls='--', color=get_color(6)) 
