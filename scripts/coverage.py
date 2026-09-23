import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import datetime as dt
from astropy.coordinates import SkyCoord
import astropy.units as u
from pathlib import Path
import argparse

from qubic.lib.Instrument.Qacquisition import QubicAcquisition
from qubic.lib.Instrument.Qinstrument import QubicInstrument
from qubic.lib.Qdictionary import qubicDict
from qubic.lib.Qscene import QubicScene
from qubic.lib.Qsamplings import QubicSampling
from qubicpack.qubicfp import qubicfp

def calculate_coverage(samplings):
    """
    Compute the QUBIC coverage map

    Parameters
    ----------
    samplings : QubicSampling
        The sampling object for the observation.
    plot : bool, optional
        If True, produce coverage plots.
    title : str, optional
        Additional title label for the plots.

    Returns
    -------
    coverage : numpy.ndarray
        Array representing the coverage map.
    seenpix : numpy.ndarray
        Boolean mask of pixels with significant coverage (>10% of max).
    """
    # Load QUBIC dictionary from file
    dictfilename = "td.dict"
    d = qubicDict()
    d.read_from_file(dictfilename)

    # Instantiate instrument, scene, and acquisition
    qubic = QubicInstrument(d)
    scene = QubicScene(d)
    acq = QubicAcquisition(qubic, samplings, scene, d)

    # Compute coverage map
    coverage = acq.get_coverage()
    seenpix = coverage > 0.1*np.max(coverage)  # pixels with >10% of max coverage


def plot_coverage(coverage, title=None):
    # Determine title prefix
    if title is None:
        title_label = 'Coverage'
    else:
        title_label = f'Coverage {title}'

    # Plot coverage in galactic coordinates
    hp.mollview(coverage, title=f'{title_label} (Galactic)', cmap='Spectral_r')

    # Overlay equatorial plane (Dec = 0°) in galactic projection
    ra = np.linspace(0, 360, 1000)
    dec_eq = np.zeros_like(ra)
    coords_eq = SkyCoord(ra=ra*u.degree, dec=dec_eq*u.degree, frame='icrs')
    gal_eq = coords_eq.galactic
    l_eq = np.radians(gal_eq.l.wrap_at(360*u.degree))
    b_eq = np.radians(gal_eq.b)
    theta_eq = np.pi/2 - b_eq.to(u.rad).value
    hp.projplot(theta_eq, l_eq, color='black', linewidth=1.5)

    # Overlay zenith line (~Dec = -23°) in galactic projection
    dec_zen = np.full_like(ra, -23.0)
    coords_zen = SkyCoord(ra=ra*u.degree, dec=dec_zen*u.degree, frame='icrs')
    gal_zen = coords_zen.galactic
    l_zen = np.radians(gal_zen.l.wrap_at(360*u.degree))
    b_zen = np.radians(gal_zen.b)
    theta_zen = np.pi/2 - b_zen.to(u.rad).value
    hp.projplot(theta_zen, l_zen, color='white', linewidth=1.5)

    plt.savefig(f'{title}_coverage.pdf', bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate coverage for a given observation date and dataset keyword.")
    parser.add_argument('date', type=str, help="Observation date in YYYY-MM-DD format")
    parser.add_argument('keyword', type=str, help="Glob pattern to filter dataset directories.")
    return parser.parse_args()


    
if __name__ == '__main__':
    args = parse_args()

    data_path = Path('./data/test_hwp/') / args.date
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    data_dirs = sorted([p for p in data_path.glob(args.keyword) if p.is_dir()])

    if not data_dirs:
        raise ValueError(f"No directories found in {data_path} matching {args.keyword}")

    print(f"Found {len(data_dirs)} datasets in {data_path}:")
    for d in data_dirs:
        print(f"{d.name}")

    coverages = []
    for dataset in data_dirs:
        a = qubicfp()
        a.read_qubicstudio_dataset(str(dataset))

        title = name

        az = a.azimuth()
        el = a.elevation()
        t = a.timeaxis('platform')
        times = t - t[0]
        date_obs = dt.datetime.utcfromtimestamp(t[0]).strftime('%Y-%m-%d %H:%M:%S')

        samplings = QubicSampling(azimuth=az, elevation=el, time=times, date_obs=date_obs, longitude=-66.46666666666667, latitude=-24.183333333333334, fix_az=False)

        coverage = calculate_coverage(samplings)[0]
        plot_coverage(coverage, title=f'{dataset.name}')
        coverages.append(coverage)

    total_coverage = np.sum(np.array(coverages), axis=0)
    plot_coverage(total_coverage, title=f'{args.date}')

    print("Done!")
