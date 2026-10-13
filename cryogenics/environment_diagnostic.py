import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from qubic_sys.hk_utilities import download_groups, read_hk_file, HK_QUANTITIES

from my_themes import set_theme, get_color, apply_theme
from plot_functions import plot_flags, on_click
set_theme('publication')

download_groups(['environment'])

start_date = np.datetime64('2026-09-20 00:00:00')
#end_date = np.datetime64('2026-10-06 10:00:00')
end_date = np.datetime64('now')
data = {}

# Pressure inside the cryostat
dat = read_hk_file(HK_QUANTITIES['environment']['pressure'])
rangemask = (dat[0] >= start_date.astype(float)) & (dat[0] <= end_date.astype(float))
time = dat[0][rangemask]

data['pressure'] = {
    'pressure': dat[1].ravel()[rangemask],
    'time': dat[0][rangemask],
    'date': (time * 1e3).astype('datetime64[ms]'),
}

# External wheater
dat = read_hk_file(HK_QUANTITIES['environment']['weather'])
rangemask = (dat[0] >= start_date.astype(float)) & (dat[0] <= end_date.astype(float))
time = dat[0][rangemask]

data['weather'] = {
    'temperature': dat[1][:,0][rangemask],
    'humidity': dat[1][:,1][rangemask],
    'pressure': dat[1][:,2][rangemask],
    'windspeed': dat[1][:,3][rangemask],
    'winddir': dat[1][:,4][rangemask],
    'time': time,
    'date': (time * 1e3).astype('datetime64[ms]'),
}

# Internal wheater
dat = read_hk_file(HK_QUANTITIES['environment']['weather_in'])
rangemask = (dat[0] >= start_date.astype(float)) & (dat[0] <= end_date.astype(float))
time = dat[0][rangemask]

data['weather_in'] = {
    'temperature': dat[1][:,0][rangemask],
    'humidity': dat[1][:,1][rangemask],
    'time': time,
    'date': (time * 1e3).astype('datetime64[ms]'),
}

# Cryostat temperature
dat = read_hk_file(HK_QUANTITIES['environment']['cryostat'])
rangemask = (dat[0] >= start_date.astype(float)) & (dat[0] <= end_date.astype(float))
time = dat[0][rangemask]

data['cryostat'] = {
    'temperature': dat[1].ravel()[rangemask],
    'time': dat[0][rangemask],
    'date': (time * 1e3).astype('datetime64[ms]'),
}



# Temperatures
plt.ion()
fig_temp, ax_temp = plt.subplots()
fig_temp.autofmt_xdate()

ax_temp.plot(data['weather']['date'], data['weather']['temperature'], marker='.', ls='none', label='Outside temperature')
ax_temp.plot(data['weather_in']['date'], data['weather_in']['temperature'], marker='.', ls='none', label='Inside temperature')
ax_temp.plot(data['cryostat']['date'], data['cryostat']['temperature'], marker='.', ls='none', label='Cryostat temperature')

ax_temp.set_ylabel('Temperature [K]')
ax_temp.set_xlabel('Date')
plt.legend()


# Outside pressure
plt.ion()
fig_pressure_out, ax_pressure_out = plt.subplots()
fig_pressure_out.autofmt_xdate()

ax_pressure_out.plot(data['weather']['date'], data['weather']['pressure'], marker='.', ls='none', label='Outside pressure')

ax_pressure_out.set_ylabel('Pressure [K]')
ax_pressure_out.set_xlabel('Date')
plt.legend()

# Inside cryostat pressure
plt.ion()
fig_pressure_in, ax_pressure_in = plt.subplots()
fig_pressure_in.autofmt_xdate()

ax_pressure_in.plot(data['pressure']['date'], data['pressure']['pressure'], marker='.', ls='none', label='Inside cryostat pressure')

ax_pressure_in.set_ylabel('Pressure [mbar]')
ax_pressure_in.set_xlabel('Date')
plt.legend()


# Humidity
plt.ion()
fig_hum, ax_hum = plt.subplots()
fig_hum.autofmt_xdate()

ax_hum.plot(data['weather']['date'], data['weather']['humidity'], marker='.', ls='none', label='Outside humidity')
ax_hum.plot(data['weather_in']['date'], data['weather_in']['humidity'], marker='.', ls='none', label='Inside humidity')

ax_hum.set_ylabel('Humidity')
ax_hum.set_xlabel('Date')
plt.legend()


# Windspeed
plt.ion()
fig_wind, ax_wind = plt.subplots()
fig_wind.autofmt_xdate()

ax_wind.plot(data['weather']['date'], data['weather']['windspeed'], marker='.', ls='none', label='Windspeed')

ax_wind.set_ylabel('Windspeed')
ax_wind.set_xlabel('Date')
plt.legend()
