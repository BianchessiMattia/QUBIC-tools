import argparse
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_body, angular_separation

#date = '2026-08-27T22:55'
#azimuth = 15
#elevation = 30

parser = argparse.ArgumentParser(description="Sun angular distance from telescope pointing")
parser.add_argument("date")
parser.add_argument("azimuth", type=float)
parser.add_argument("elevation", type=float)

args = parser.parse_args()

SITE = EarthLocation(lat=-24.18333 * u.deg, lon=-66.466666*u.deg)
time = Time(args.date)
telescope_frame = AltAz(obstime=time, location=SITE)
telescope = SkyCoord(az=args.azimuth*u.deg, alt=args.elevation*u.deg, frame=telescope_frame)
sun = get_body('sun', time, SITE)
dist = telescope.separation(sun)
print(f'Sun angular distance: {dist.deg}')
