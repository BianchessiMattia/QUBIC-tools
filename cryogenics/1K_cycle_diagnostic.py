import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import polars as pl

from qubic_sys.hk_utilities import download_groups, read_hk_file, HK_QUANTITIES, plot_temperature

from my_themes import set_theme, get_color, apply_theme
from plot_functions import plot_flags, on_click
set_theme('publication')

download_groups(['1K'])

start_date = np.datetime64('2026-10-01 00:00:00')
#end_date = np.datetime64('2026-09-30 23:59:00')
end_date = np.datetime64('now')

data = {}
for quantity in HK_QUANTITIES['1K']:
    print('Loading', quantity)
    dat = read_hk_file(HK_QUANTITIES['1K'][quantity])

    rangemask = (dat[0] >= start_date.astype(float)) & (dat[0] <= end_date.astype(float))
    temperature = dat[1][rangemask]
    time = dat[0][rangemask]
    date = (time * 1e3).astype('datetime64[ms]')

    data[quantity] = {
        'temperature': temperature,
        'time': time,
        'date': date,
    }


### Plot
fig,ax = plot_temperature(data, '1K fridge cycles')

plt.savefig(f'figures/1K_fridge_cycles_{end_date}.png', dpi=300)

plt.connect('button_press_event', on_click)

ax.axvline(np.datetime64('2026-10-08 18:10:00'), ls='--', color=get_color(6))
#ax.axhline(1.2, ls='--', color=get_color(6))

#plot_flags(ax, {np.datetime64('2026-09-23 17:00:00'): 'compressors off'})


### Cycles
cycles = pl.read_csv('cycles_1Kstage.csv', separator="\t", has_header=True, infer_schema=False, truncate_ragged_lines=True)

starts = cycles["start_utc"].to_numpy().astype('datetime64[s]')
ends = cycles["end_utc"].to_numpy().astype('datetime64[s]')

missing = np.isnat(ends)
ends[missing] = starts[missing] + np.timedelta64(4, "h")
cycles = cycles.with_columns(pl.Series("end_utc", np.datetime_as_string(ends, unit="s")))
cycles.write_csv('cycles_1Kstage.csv', separator="\t", null_value="")

ax.axhline(1.2, ls='--', lw=2, color=get_color(7))
for start in starts[(starts>start_date) & (starts<end_date)]:
    ax.axvline(start, ls='--', lw=2, color=get_color(6))
for end in ends[(ends>start_date) & (ends<end_date)]:
    ax.axvline(end, ls='--', lw=2, color=get_color(6))
