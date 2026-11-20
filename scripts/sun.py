import argparse
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body, angular_separation

date = '2026-11-18T12:29'
azimuth = 124
elevation = 32.5

SITE = EarthLocation(lat=-24.18333 * u.deg, lon=-66.466666*u.deg)
time = Time(date)
telescope_frame = AltAz(obstime=time, location=SITE)
telescope = SkyCoord(az=azimuth*u.deg, alt=elevation*u.deg, frame=telescope_frame)
sun = get_body('sun', time, SITE)
dist = telescope.separation(sun)
print(f'Sun angular distance: {dist.deg}')
