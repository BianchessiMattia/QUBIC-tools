import os 
import polars as pl 
import numpy as np
import matplotlib.pyplot as plt

from qubicpack.housekeeping.utilities import download_hk

SAVE_DIR = '/home/mattia/se/hk/2026/'

HK_QUANTITIES = {
    '1K': {
        '1K_stage': 'AVS47_1_ch1', 
        '1K_CH': 'AVS47_1_ch4',
        '1K_CP': 'TEMPERATURE18', 
        '1K_HS': 'TEMPERATURE17',
    },
    '300mK': {
        'tes_stage': 'AVS47_1_ch2', 
        '300mK_CH': 'AVS47_1_ch6',
        '300mK_4HS': 'TEMPERATURE14', 
        '300mK_3HS': 'TEMPERATURE16',
        '300mK_4CP': 'TEMPERATURE13', 
        '300mK_3CP': 'TEMPERATURE15',
    },
    'pt': {
        'PT2_S2_CH': 'AVS47_2_ch0', 
        'PT1_S2_CH': 'AVS47_2_ch1',
        'PT2_S1': 'TEMPERATURE04', 
        #'PT1_S1': 'TEMPERATURE05',
        'PT2_S2': 'TEMPERATURE12', 
        'PT1_S2': 'TEMPERATURE11',
        'fridge_plate': 'AVS47_2_ch2', 
        '4K_shield': 'AVS47_2_ch4',
    },
    'environment': {
        'pressure': 'PRESSURE1', 
        'weather': 'weather',
        'weather_in': 'inside_weather', 
        'cryostat': 'CRYOSTAT',
    },
}


def download_groups(groups):
    """
    Download the housekeeping files of the requested groups.

    Parameters
    ----------
    groups : list of str
        Group names from HK_QUANTITIES, or ['all'].
    """
    if 'all' in groups:
        groups = list(HK_QUANTITIES)

    files = []
    for group in groups:
        if group not in HK_QUANTITIES:
            raise ValueError(f"Unknown group {group}. Available: {list(HK_QUANTITIES)}")
        files.extend(HK_QUANTITIES[group].values())

    download_hk(files, SAVE_DIR)


def read_hk_file(filename):
    '''
    Read timestamp and data from a Housekeeping broadcast file.

    Parameters
    ----------
    filename : str
        Path to the housekeeping file.

    Returns
    -------
    timestamps : ndarray
        Unix timestamps.
    values : ndarray
        Housekeeping values.
    '''
    filename = filename + '.txt'
    filename = SAVE_DIR + filename
    if not os.path.isfile(filename):
        print('ERROR! File not found: %s' % filename)
        return None, None

    df = pl.read_csv(filename, separator=' ', has_header=False, ignore_errors=True, truncate_ragged_lines=True)

    timestamps = df[:,0].to_numpy()
    values = df.select(df.columns[1:]).to_numpy()

    return timestamps, values


def plot_temperature(data, title=None):
    plt.ion()
    fig, ax = plt.subplots()
    fig.autofmt_xdate()

    for t in data.keys():
        ax.plot(data[t]['date'], data[t]['temperature'], marker='.', ls='none', label=t)

    ax.set_ylim(0.3,40)
    ax.set_ylabel('Temperature [K]')
    ax.set_xlabel('Date')
    plt.legend(loc='upper left')
    if title:
        plt.title(title)

    return fig, ax


def load_hk(hk_name, start, end):
    """
    Read a housekeeping file and keep only the samples inside [start, end].

    Parameters
    ----------
    hk_name : str
        HK file stem, e.g. HK_QUANTITIES['1K']['1K_stage'].
    start, end : numpy.datetime64
        Boundaries of the time window (inclusive).

    Returns
    -------
    dict
        {'temperature', 'time' (unix), 'date' (datetime64[ms])} restricted to the window.
    """
    timestamps, values = read_hk_file(hk_name)

    mask = (timestamps >= start.astype(float)) & (timestamps <= end.astype(float))
    time = timestamps[mask]

    return {
        'temperature': values[mask],
        'time': time,
        'date': (time * 1e3).astype('datetime64[ms]'),
    }


def load_fridge_cycles(csv_path, start, end):
    """
    Read a fridge-cycle table and keep the cycles overlapping [start, end].

    A cycle is kept if it starts before `end` and finishes after `start`,
    so cycles only partially inside the window are included.

    Parameters
    ----------
    csv_path : str
    start, end : numpy.datetime64
        Boundaries of the time window.

    Returns
    -------
    starts, ends : ndarray of datetime64[s]
        Start and end times of the selected cycles.
    """
    cycles = pl.read_csv(csv_path, separator="\t", has_header=True, infer_schema=False, truncate_ragged_lines=True)
    starts = cycles["start_utc"].to_numpy().astype('datetime64[s]')
    ends = cycles["end_utc"].to_numpy().astype('datetime64[s]')

    overlap = (starts < end) & (ends > start)
    return starts[overlap], ends[overlap]

