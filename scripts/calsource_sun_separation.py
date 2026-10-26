"""
calsource_sun_separation.py
Author: Mattia Bianchessi
Email: mbianchess@apc.in2p3.fr

Compute the angular separation between the Sun and the telescope pointing at fixed elevation for a range of azimuths, used to check if it is possibile to scan the calibration source during the day. Produces a separation-vs-time plot and a polar plot of the Sun trajectory against the pointing directions.
"""

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body
import numpy as np
import matplotlib.pyplot as plt

from my_themes import set_theme
set_theme('terminal')

# Observing site (Alto Chorrillos)
SITE = EarthLocation(lat=-24.18333 * u.deg, lon=-66.466666*u.deg)

# Time grid: 7 hours starting at start_time, sampled every 5 minutes
start_time = Time('2026-10-23 13:00:00')
time = start_time + np.arange(0,7*60,5) * u.min

# Sun position as a function of time, expressed in the local Alt-Az frame
telescope_frame = AltAz(obstime=time, location=SITE)
sun = get_body('sun', time, SITE)
sun_altaz = sun.transform_to(telescope_frame)

azimuth = np.arange(-5,25,2)
elevation = 30
dist = []
for az in azimuth:
    telescope = SkyCoord(az=az*u.deg, alt=elevation*u.deg, frame=telescope_frame)
    dist.append(telescope.separation(sun))


# Figure 1: angular separation vs time, one curve per azimuth
plt.ion()
fig, ax = plt.subplots()
fig.autofmt_xdate()
for i in range(len(azimuth)):
    ax.plot(time.value.astype('datetime64[s]'), dist[i], label=f'Azimuth = {azimuth[i]}°')
ax.axhline(30, ls='--', color='red')
ax.axhline(50, ls='--', color='red')
ax.set_xlabel('Time [UTC]')
ax.set_ylabel('Angular separation')
ax.set_title(f'Sun - Telescope angular separation / elevation = {elevation}')
ax.legend()
plt.savefig(f'./sun_angular_separation_elevation_{elevation}.pdf', bbox_inches='tight')


# Figure 2: polar Alt-Az plot (azimuth from North, clockwise; zenith at center)
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
ax.set_rlim(90, 0)

# Telescope pointings (fixed elevation) and Sun track over the time window
ax.scatter(np.deg2rad(azimuth), np.full(len(azimuth), elevation), label='Telescope')
ax.scatter(np.deg2rad(sun_altaz.az.deg), sun_altaz.alt.deg, marker='*', c='gold', label='Sun')

data_str = time[0].iso.split(" ")[0]
start_hour_str = time[0].iso.split(" ")[1][0:5]
end_hour_str = time[-1].iso.split(" ")[1][0:5]
ax.set_title(f"Sun trajectory\n Date: {data_str}\nfrom {start_hour_str} to {end_hour_str}")
ax.legend()
