"""
rotation_time_statistics.py 

Author: Mattia Bianchessi
Email: mbianchessi@apc.in2p3.fr
"""

from astropy.table import Table
import numpy as np
import argparse
from pathlib import Path
from astropy.io.ascii import convert_numpy

parser = argparse.ArgumentParser()
parser.add_argument(
    "log_path",
    type=Path,
    )
args = parser.parse_args()

converters = {}
converters["start_time"] = [convert_numpy(str)]
converters["end_time"] = [convert_numpy(str)]

table = Table.read(args.log_path, format="csv", converters=converters)
date = f'{str(args.log_path.name)[0:4]}-{str(args.log_path.name)[4:6]}-{str(args.log_path.name)[6:8]}'

complete = table[table["outcome"] != "incomplete"]
start_time = [int(t[0:2])*3600 + int(t[2:4])*60 + int(t[4:6]) for t in complete["start_time"]]
end_time = [int(t[0:2])*3600 + int(t[2:4])*60 + int(t[4:6]) for t in complete["end_time"]]
complete["rotation_time"] = np.array(end_time) - np.array(start_time)

success = complete[complete["outcome"] == "success"]
ok = success["rotation_time"][success["rotation_time"] != 0]

failure = complete[complete["failure_reason"] == "engage_failed"]
fail = failure["rotation_time"][failure["rotation_time"] != 0]

print(f'{date},{ok.size},{np.mean(ok):.2f},{np.median(ok):.2f},{np.std(ok):.2f},{fail.size},{np.mean(fail):.2f},{np.median(fail):.2f},{np.std(fail):.2f}')
