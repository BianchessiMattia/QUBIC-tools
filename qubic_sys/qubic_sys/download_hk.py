#!/usr/bin/env python3
"""
download_hk.py

Author: Mattia Bianchessi
Email: mbianchessi@apc.in2p3.fr
Created: 2026-10-01

Download QUBIC housekeeping files from the qubic-central.
"""
import argparse
import subprocess

from qubic_sys.hk_utilities import download_groups

def main():
    parser = argparse.ArgumentParser(description="Download QUBIC housekeeping files.")
    parser.add_argument('groups', nargs='*', default=['all'],
                        help="group names or 'all' (default: all)")
    args = parser.parse_args()

    download_groups(args.groups)
    subprocess.run('rsync -Pazt qubic:/home/qubic/data/temperature/Cycles_1Kfridge4_Nov2026.log /home/mattia/se/cryogenics/', shell=True)
    subprocess.run('rsync -Pazt qubic:/home/qubic/data/temperature/Cycles_300mKfridge_Nov2026.log /home/mattia/se/cryogenics/', shell=True)


if __name__ == '__main__':
    main()
