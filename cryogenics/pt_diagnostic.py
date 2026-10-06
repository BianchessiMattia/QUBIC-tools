import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from qubic_sys.hk_utilities import download_groups, read_hk_file, HK_QUANTITIES

from my_themes import set_theme, get_color, apply_theme
from plot_functions import plot_flags, on_click
set_theme('publication')

download_groups(['pt'])

start_date = np.datetime64('2026-09-20 00:00:00')
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

plt.ion()
fig, ax = plt.subplots()
fig.autofmt_xdate()

for t in data.keys():
    ax.plot(data[t]['date'], data[t]['temperature'], marker='.', ls='none', label=t)

ax.set_ylim(2.5,20)
ax.set_ylabel('Temperature [K]')
ax.set_xlabel('Date')
plt.legend(loc='upper left')
plt.title('Pulse Tube temperature diagnostic')

plt.savefig('1.pdf', bbox_inches='tight')

plt.connect('button_press_event', on_click)

ax.axvline(np.datetime64('2026-09-23 13:50:00'), ls='--', color=get_color(6)) 
