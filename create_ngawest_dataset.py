"""
Template for the generation of a Time histories' database. Steps for the generation
are reported in the README file and here below,
**but please refer to the README for the most up-to-date source**.

Copy create_dataset.py as well as metadata_fields.yml in a empty directory

Edit your source metadata file (CSV format) to match the field names in
metadata_fields.yml. You can also start from metadata_template.csv as empty template,
leaving empty cells if data is N/A or missing, or you plan to fill it inside
create_dataset.py

Edit create_dataset.py

3a. Set the path of the source metadata file (variable source_metadata_path)

3b. Implement how to read time histories from the metadata file rows
    (functions get_waveforms_path and read_waveform)

3c. Implement how to process time histories and potentially modify the associated CSV row
    (function process_waveforms)

Eventually, execute create_dataset.py file on the terminal within the Python virtual
environment (or Conda env):

python3 create_dataset.py

The file will scan all rows of your source metadata file, process them and put them in
the waveforms subdirectory of the root directory of create_dataset.py.
A new metadata file metadata.csv will be also created in the same directory
"""
from __future__ import annotations

from typing import Optional, Any, Tuple
import os
from os.path import abspath, join, basename, isdir, isfile, dirname, splitext
import stat
import yaml
import re
import h5py
import csv
import json
import sys
import fnmatch
import obspy
from numpy import ndarray
# from obspy import Trace, Stream
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import scipy
import glob
from tqdm import tqdm

########################################################################
# Editable file part: please read carefully and implement your routine #
########################################################################

source_metadata_path: str = "path/to/my/source/metadata.csv"  # Your source metadata CSV file

source_metadata_csv_args = {
    'header': None}  # csv arguments for source metadata (e/g. 'header'= None)


def get_waveforms_path(metadata: dict) -> tuple[
    Optional[str], Optional[str], Optional[str]]:
    """Get the full source paths of the waveforms (h1, h2 and v components, in this
    order) from the given row of your source metadata.
    Paths can be empty or None, meaning that the relative file is missing. This has to
    be taken into account in `process_waveforms` in case (see below). If files are not
    missing, then the file must exist

    :param metadata: Python dict, corresponding to a row of your source metadata table.
        Each dict key represents a Metadata Field (Column). Note that float, str,
        datetime and categorical values can also be None (e.g., if the Metadata cell
        was empty)
    """
    # get the path stored in the metadata file (just an example):
    h1_path = metadata['fpath_h1']
    h2_path = metadata['fpath_h2']
    v_path = metadata['fpath_v']

    # Example code (please MODIFY) given a source dir of raw time histories:
    source_root_dir = "my/source/root"

    # Option 1 (simple), simple concatenation of the source metadata fields:
    files = (
        join(source_root_dir, h1_path),
        join(source_root_dir, h2_path),
        join(source_root_dir, v_path)
    )

    # Option 2 (slightly more complex, files are nested inside source root dir):
    files = ["", "", ""]
    for idx, relative_path in [h1_path, h2_path, v_path]:
        for dirpath, dirnames, filenames in os.walk(source_root_dir):
            candidate = os.path.join(dirpath, relative_path)
            if isfile(candidate):
                files[idx] = os.path.abspath(candidate)
                break

    # Return the files. We do a last check setting a path to None if it does not exist,
    # to signal the routine that the file should not be read. If you remove the line
    # below, files must exist and the routine will break in case
    return (
        None if not isfile(files[0]) else files[0],
        None if not isfile(files[1]) else files[1],
        None if not isfile(files[2]) else files[2]
    )

def read_waveform(full_abs_path: str, metadata_row:dict) -> tuple[float, ndarray]:
    """Read a waveform from a file path. Modify according to the format you stored
    your time histories"""
    with open(full_abs_path) as f:
        # First few lines are headers
        header1 = f.readline().strip()
        header2 = f.readline().strip()
        header3 = f.readline().strip()
        header4 = f.readline().split()
        # ndat = int(header4[1][:-1])
        # dtsamp = float(header4[3])
        ### By using the small following routine, I am finding only the data with digits from this header4 line.
        # numeric = '0123456789-.'
        # numbers = []
        # for h in header4:
        #     for i, c in enumerate(h + ' '):
        #         if c not in numeric:
        #             break
        #     if len(h[:i]) > 0:
        #         digits = h[:i]
        #         numbers.append(digits)
        # # In the numbers only first and second ndat and dtsamp.
        # ndat = int(numbers[0])
        # dtsamp = float(numbers[1])
        # times = np.array([dtsamp * j for j in range(ndat)])
        reg = re.match("^NPTS\s*=\s*(\d+)\s*,\s*DT\s*=\s*([\.\d]+)\s*SEC,*\s*$")
        dt = float(reg.group(1))
        # data = []
        # for line in f:
        #     line = line.strip().split()
        #     yy = []
        #     for l in line:
        #         try:
        #             yy.append(float(l))
        #         except ValueError:
        #             yy.append(l)
        #     data.extend(yy)
        data = np.fromiter((float(n) for line in f for n in line), dtype=float)
        f.close()

        # get arrival time and

        # The acceleration time series is given in units of g. So I convert it in m/s.
        # However it is said only that its in the units of g, not sure if its 980 cm/s^ or 9.8 m/s^2
        # g = 9.8  # in (m/s^2)
        # data = np.array([number * g for number in data])
        # pga = max([abs(number) for number in data])
        # pga = max(np.abs(data))
        # pga_ind = [abs(number) for number in data].index(pga)
        # pga_ind = np.argmax(np.abs(data))
        return dt, data


def process_waveforms(
        metadata: dict,
        h1: tuple[float, ndarray],
        h2: tuple[float, ndarray],
        v: tuple[float, ndarray]
) -> tuple[
    dict,
    tuple[float, ndarray] | None,
    tuple[float, ndarray] | None,
    tuple[float, ndarray] | None
]:
    """Process the waveform(s), returning the same argument modified according to your
    custom processing routine: a new metadata dict, and three obspy Traces denoting the
    processed two horizontal and vertical components, respectively.

    The metadata  dict can be built from `metadata_row` or from scratch, it must
    include all fields defined in `metadata_fields.yml`. fpath* fields might be renamed
    to supply the destination path of the traces, relative to the CSV file. Other fields
    must be input with the correct data type: note that float, str, datetime and
    categorical fields can be filled with None to indicate that the corresponding value
    is missing or unknown.

    For time histories, please remember to provide waveforms in standard units
    (m/sec*sec, m/sec, m) and consistent with the value of of 'sensor_type' in the
    returned metadata dict ('A', 'V', 'D'). Remember that, depending on your use-case,
    some Traces might be None.

    :param metadata: Python dict, corresponding to a row of your source metadata table.
        Each dict key represents a Metadata Field (Column). Note that float, str,
        datetime and categorical values can also be None (e.g., if the Metadata cell
        was empty)
    :param h1: first horizontal component, as tuple(dt:float, data:numeric_array),
        or None (no Trace)
    :param h2: second horizontal component, same format as h1
    :param v: vertical component, same format as h1
    """
    # compute arrival time (correct)
    evt_time = datetime.fromisoformat(metadata.get(13))
    evt_depth_km = metadata.get(33)
    r_epi_km = metadata.get(55)

    # re-compute the time series start time

    # # Get first arrival for P-phase
    # arrivals = tt_model.get_travel_times(
    #     source_depth_in_km=evt_depth_km,
    #     distance_in_degree=kilometer2degrees(r_epi_km),
    #     phase_list=["P"]
    # )
    #
    # # Extract travel time in seconds (float)
    # t_time = 0  # default trqvel time (from event origin time) is assumed to be 0 seconds
    # if arrivals:
    #     t_time = arrivals[0].time  # in seconds
    #
    # t_time -= 450  # account for the nga-west time window before P-arrival, loosely
    # # estimated from the figures at page 22 of:
    # # https://peer.berkeley.edu/sites/default/files/webpeer-2014-17-christine_a._goulet_tadahiro_kishida_timothy_d._ancheta_chris_h._cramer_robert_b._final.pdf)  # noqa
    #
    # # modify the traces with the correct arrival time:
    # h1.stats.starttime = evt_time + timedelta(seconds=t_time)
    # h2.stats.starttime = evt_time + timedelta(seconds=t_time)
    # v.stats.starttime = evt_time + timedelta(seconds=t_time)

    evt_id = metadata.get(12)
    new_metadata = {
        'azimuth': metadata.get(62),
        'repi': metadata.get(55),
        'rrup': metadata.get(60),
        'rhypo': metadata.get(56),
        'rjb': metadata.get(57),
        'rx': None,
        'ry': None,
        'rvolc': None,

        'evt_id': evt_id,
        'evt_time': evt_time,
        'evt_lat': metadata.get(31),
        'evt_lon': metadata.get(32),
        'mag': metadata.get(14),
        'mag_type': metadata.get(15),
        'evt_depth': metadata.get(33),
        'rup_top_depth': metadata.get(39),
        'rup_width': metadata.get(41),
        'strike': metadata.get(23),
        'dip': metadata.get(24),
        'rake': metadata.get(25),
        'sof': metadata.get(26),

        'sta_id': metadata.get(11) if metadata.get(11) != "99999" else
        "#" + metadata.get(10),
        "z1": metadata.get(105),
        "z2pt5": metadata.get(107),
        "vs30": metadata.get(87),
        "backarc": False,  # FIXME check!
        "sta_lat": metadata.get(95),
        "sta_lon": metadata.get(96),
        "vs30measured": metadata.get(89) in {0, "0", 0.0},
        # "xvf": None,
        "region": 0,
        "fpeak": None,
        # "geology": None,
        "sensor_type": 'A',
        "fpath": f"{evt_id}/{splitext(basename(metadata.get(116)))[0] + '.h5'}",
        # "fpath_h1": f"{evt_id}/{metadata.get(116)}" if metadata.get() else None,
        # "fpath_h2": f"{evt_id}/{metadata.get(117)}" if metadata.get() else None,
        # "fpath_v": f"{evt_id}/{metadata.get(118)}" if metadata.get() else None,
        "filter_type": metadata.get(123),

        "npass": metadata.get(124),
        "nroll": metadata.get(125),
        "hp_h1": metadata.get(126),
        "hp_h2": metadata.get(127),
        "lp_h1": metadata.get(128),
        "lp_h2": metadata.get(129),
        "luf_h1": metadata.get(131),
        "luf_h2": metadata.get(132)
    }
    # simply return the arguments (no processing by default):
    return new_metadata, h1, h2, v


###########################################
# The code below should not be customized #
###########################################


def save_waveforms(
        root_path,
        metadata: dict,
        h1: tuple[float, ndarray],
        h2: tuple[float, ndarray],
        v: tuple[float, ndarray]
):

    file_path = abspath(join(root_path, metadata['fpath']))
    os.makedirs(file_path)

    with h5py.File(file_path, "w") as f:
        # Save existing components
        if h1 is not None:
            f.create_dataset("h1", data=h1[1])
            f["h1"].attrs["dt"] = h1[0]

        if h2 is not None:
            f.create_dataset("h2", data=h2[1])
            f["h2"].attrs["dt"] = h2[0]

        if v is not None:
            f.create_dataset("v", data=v[1])
            f["v"].attrs["dt"] = v[0]

    # Add read permission for group (stat.S_IRGRP) and others (stat.S_IROTH).
    os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IRGRP | stat.S_IROTH)

#     if h1 is not None:
#         save_waveform(h1, abspath(join(root_path, record['fpath_h1'])))
#     if h2 is not None:
#         save_waveform(h2, abspath(join(root_path, record['fpath_h2'])))
#     if v is not None:
#         save_waveform(v, abspath(join(root_path, record['fpath_v'])))
#
#
# def save_waveform(trace, dest_path):
#     os.makedirs(os.path.dirname(dest_path))
#     Stream([trace]).write(dest_path, format='MSEED')
#     # Add read permission for group (stat.S_IRGRP) and others (stat.S_IROTH).
#     os.chmod(
#         dest_path,
#         os.stat(dest_path).st_mode | stat.S_IRGRP | stat.S_IROTH
#     )


def _cast_dtype(val: Any, dtype: str):
    if dtype == 'float':
        assert val is None or isinstance(val, float)
    elif dtype == 'int':
        assert isinstance(val, int)
    elif dtype == 'bool':
        if val in {0, 1}:
            val = bool(val)
        assert isinstance(val, bool)
    elif dtype == 'datetime':
        if hasattr(val, 'to_pydatetime'):  # for safety
            val = val.to_pydatetime()
        assert val is None or isinstance(val, datetime)
    elif val == 'str':
        assert val is None or isinstance(val, str)
    elif isinstance(dtype, (list, tuple)):
        assert val is None or val in dtype
    return val


if __name__ == "__main__":

    dest_root_path = dirname(abspath(__file__))
    dest_metadata_path = join(dest_root_path, "metadata.csv")
    dest_waveforms_path = join(dest_root_path, "waveforms")

    existing = isfile(dest_metadata_path) or isdir(dest_waveforms_path)
    if existing:
        res = input(
            f'Metadata file ({basename(dest_metadata_path)}) or waveforms dir '
            f'({basename(dest_waveforms_path)}) already exist in {dest_root_path}.\n'
            f'If you type "y", Metadata file will be deleted and recreated, and '
            f'waveforms files potentially overwritten.\n'
            f'Proceed (y=yes, any key=no)?'
        )
        if res != 'y':
            sys.exit(1)

    if isfile(dest_metadata_path):
        os.unlink(dest_metadata_path)

    # sanitize the metadata using associated yaml:
    metadata_fields: dict = yaml.safe_load(join(dest_root_path, "metadata_fields.yml"))

    smp = source_metadata_path
    with open(smp, 'rb') as f:
        max_rows = sum(1 for _ in f) - 1  # Subtract 1 for header

    pbar = tqdm(
        total=max_rows,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "(estimated remaining time {remaining}s)"
    )

    rec_num = 0
    csv_args = dict(source_metadata_csv_args)
    csv_args.setdefault('chunksize', 100000)
    for i, metadata in pd.read_csv(smp, **csv_args):
        new_metadata = []
        for record in metadata.itertuples(index=False):
            rec_num += 1
            metadata_row = f'metadata row #{rec_num}'
            pbar.update(1)

            record = record._asdict()
            try:
                h1, h2, v = get_waveforms_path(record)
            except Exception as exc:
                print(
                    f"Error in get_waveforms_path ({metadata_row}): {exc}",
                    file=sys.stderr
                )
                sys.exit(1)

            # read waveforms separately:
            try:
                h1 = None if h1 is None else read_waveform(h1, metadata)
            except Exception as exc:
                print(
                    f"Error in get_waveforms_path ({metadata_row}) building "
                    f"h1 path. File not found: {h1}", file=sys.stderr
                )
                sys.exit(1)
            try:
                h2 = None if h2 is None else read_waveform(h2, metadata)
            except Exception as exc:
                print(
                    f"Error in get_waveforms_path ({metadata_row}) building "
                    f"h2 path. File not found: {h2}", file=sys.stderr
                )
                sys.exit(1)
                # read waveforms separately:
            try:
                v = None if v is None else read_waveform(v, metadata)
            except Exception as exc:
                print(
                    f"Error in get_waveforms_path ({metadata_row}) building "
                    f"v path. File not found: {v}", file=sys.stderr
                )
                sys.exit(1)

            try:
                record, h1, h2, v = process_waveforms(record, h1, h2, v)
            except Exception as exc:
                print(
                    f"Error in process_waveforms ({metadata_row}): {exc}",
                    file=sys.stderr
                )
                sys.exit(1)

            # check returned metadata
            for f in metadata_fields.items():
                dtype = metadata_fields[f]['dtype']
                try:
                    record[f] = _cast_dtype(record[f], dtype)
                except AssertionError:
                    print(
                        f"Error in {metadata_row} after processing: "
                        f"'{f}' should be {dtype}",
                        file=sys.stderr
                    )
            new_metadata.append(record)

            # save waveforms
            try:
                save_waveforms(dest_waveforms_path, record, h1, h2, v)
            except Exception as exc:
                print(
                    f"Error in save_waveforms ({metadata_row}): {exc}",
                    file=sys.stderr
                )
            sys.exit(1)

        # save metadata:
        pd.DataFrame(new_metadata).to_csv(
            dest_metadata_path,
            date_format="%Y-%m-%dT%H:%M:%S",
            index=False,
            mode='a',
            header=(i == 0),
            na_rep=''
        )

    os.chmod(
        dest_metadata_path,
        os.stat(dest_metadata_path).st_mode | stat.S_IRGRP | stat.S_IROTH
    )
    pbar.close()
    sys.exit(0)
