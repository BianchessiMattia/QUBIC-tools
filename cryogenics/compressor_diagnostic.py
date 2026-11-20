import numpy as np
import matplotlib.pyplot as plt

from qubic_sys.hk_utilities import download_groups, read_compressor_log

from my_themes import set_theme
from plot_functions import plot_flags, on_click, save_figure
set_theme('terminal')

download_groups(['compressors'])

start_date = np.datetime64('2026-11-04 00:00:00')
#end_date = np.datetime64('2026-10-06 10:00:00')
end_date = np.datetime64('now')
data = {}

for c in ['compressor1', 'compressor2']:
    dat = read_compressor_log(f'{c}_log')
    rangemask = (dat[0] >= start_date) & (dat[0] <= end_date)

    data[c] = {
        'date': dat[0][rangemask],
        'T_He': dat[1]['T_He'][rangemask],
        'T_out': dat[1]['Tout'][rangemask],
        'T_in': dat[1]['Tin'][rangemask],
        'P_in': dat[1]['Pin'][rangemask],
        'hours': dat[1]['Hours'][rangemask]
    }

plt.ion()
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
plt.connect('button_press_event', on_click)
fig.autofmt_xdate()

for compressor_number in [1,2]:
    compressor = f'compressor{compressor_number}'
    ax1.plot(data[compressor]['date'], data[compressor]['P_in'], marker='.', ls='none', label=f'Compressor {compressor_number}')
    ax2.plot(data[compressor]['date'], data[compressor]['T_in'], marker='.', ls='none', label=f'T_in compressor {compressor_number}')
    ax2.plot(data[compressor]['date'], data[compressor]['T_out'], marker='.', ls='none', label=f'T_out compressor {compressor_number}')
    ax3.plot(data[compressor]['date'], data[compressor]['T_He'], marker='.', ls='none', label=f'T_He compressor {compressor_number}')

ax1.set_title('Pulse tube compressors')
ax1.set_ylim(5.7,8)
ax1.legend()
ax1.set_ylabel('Return pressure [bar]')
ax2.set_ylim(12,24)
ax2.legend()
ax2.set_ylabel('Temperature [°C]')
ax3.legend()
ax3.set_xlabel('Date')
ax3.set_ylabel('Temperature [°C]')
ax3.set_ylim(40,56)

#save_figure('figures/compressors', x_is_date=True, extension='png', dpi=300)



'''
flags = {
    np.datetime64('2026-11-06T22:22:09') : 'First low pression warning',
    np.datetime64('2026-11-08T15:51:08') : 'Manually swithed off compressor2',
    np.datetime64('2026-11-11T15:30:00') : 'Helium refill',
    np.datetime64('2026-11-12T10:37:00') : 'Heating up cryopumps',
    np.datetime64('2026-11-12T13:40:00') : 'Leak fixed',
}
'''
