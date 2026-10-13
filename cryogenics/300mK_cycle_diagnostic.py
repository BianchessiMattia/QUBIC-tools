import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from qubic_sys.hk_utilities import download_groups, read_hk_file, HK_QUANTITIES, plot_temperature

from my_themes import set_theme, get_color, apply_theme
from plot_functions import plot_flags, on_click
set_theme('terminal')

download_groups(['300mK'])

start_date = np.datetime64('2026-09-20 00:00:00')
#end_date = np.datetime64('2026-10-06 10:00:00')
end_date = np.datetime64('now')

data = {}
for quantity in HK_QUANTITIES['300mK']:
    print('Loading', quantity)
    dat = read_hk_file(HK_QUANTITIES['300mK'][quantity])

    rangemask = (dat[0] >= start_date.astype(float)) & (dat[0] <= end_date.astype(float))
    temperature = dat[1][rangemask]
    time = dat[0][rangemask]
    date = (time * 1e3).astype('datetime64[ms]')

    data[quantity] = {
        'temperature': temperature,
        'time': time,
        'date': date,
    }

fig, ax = plot_temperature(data, '300mK fridge cycles')

plt.savefig('figures/300mK_fridge_cycles.png', dpi=300)

plt.connect('button_press_event', on_click)

ax.axvline(np.datetime64('2026-09-23 13:50:00'), ls='--', color=get_color(6))
        
