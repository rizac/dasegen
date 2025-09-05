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
import logging
import urllib.request
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
from numpy import ndarray
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm


########################################################################
# Editable file part: please read carefully and implement your routine #
########################################################################

def find_waveforms_path(metadata: dict, waveforms_path: set[str]) \
        -> tuple[Optional[str], Optional[str], Optional[str]]:
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
    candidates: dict[str, list[str]] = {
        metadata['fpath_h1']: [],
        metadata['fpath_h2']: [],
        metadata['fpath_v']: []
    }

    for metadata_path, files in candidates.items():
        if not metadata_path:
            continue
        for file_abs_path in waveforms_path:
            if metadata_path in basename(file_abs_path):
                files.append(file_abs_path)

    return tuple(  # noqa
        None if not k or len(v) != 1 else v[0] for k, v in candidates.items()
    )



def read_waveform(full_abs_path: str, metadata: dict) -> tuple[float, ndarray]:
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


# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {}

# relative error threshold. After 100 waveforms, when waveforms with error / warnings
# get higher than this number (relative to the total number of processed waveforms)
# the program will stop. 0.05 means 5% max of erroneous waveforms
re_err_th = 0.05


def get_dest_dir_path():
    """destination root path, defaults to this script dir"""
    return dirname(abspath(__file__))


def main():

    source_metadata_path = None
    source_waveforms_path = None

    if len(sys.argv) == 3:
        source_metadata_path = sys.argv[1]
        source_waveforms_path = sys.argv[2]

    err_noarg = not source_metadata_path or not source_waveforms_path
    err_no_file = not isfile(source_metadata_path) or not isdir(source_waveforms_path)

    if err_no_file or err_noarg:
        print(f'Error: {"invalid arguments" if err_noarg else "invalid file/dir path"}',
              file=sys.stderr)
        print(f"Usage: {sys.argv[0]} <metadata_table_file> <time_histories_dir>")
        print(f"Process and harmonzie an old dataset into a new one. "
              f"Takes every row of metadata_table_file (csv), finds the "
              f"relative 3 time histories inside <time_histories_dir> (recursively)"
              f"and saves the new metadata in ./metadata.csv and the new processed time"
              f"histories in ./waveforms (. refers to this script current directory)")
        sys.exit(1)

    logging.info(f'Script: {" ".join([sys.executable] + sys.argv)}')
    print(f"Source metadata path: {source_metadata_path}")
    print(f"Source waveforms path: {source_waveforms_path}")

    dest_root_path = get_dest_dir_path()
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
    print("Loading metadata fields from git repo")
    # Download and save locally
    try:
        with urllib.request.urlopen("https://raw.githubusercontent.com"
                                    "/rizac/dasegen/refs/heads/main/"
                                    "metadata_fields.yml") as response:
            metadata_fields_content = response.read()
            # Load YAML into Python dict
            metadata_fields = yaml.safe_load(metadata_fields_content.decode("utf-8"))
            # save to file
            with open(join(dest_root_path, 'metadata_fields.yml'), "wb") as f:
                f.write(metadata_fields_content)
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    with open(source_metadata_path, 'rb') as f:
        max_rows = sum(1 for _ in f) - 1  # Subtract 1 for header

    log_dest_path = dest_metadata_path + ".log"
    logging.basicConfig(
        filename=log_dest_path,
        level=logging.INFO,
        format='%(message)s'  # minimal: just the message
    )

    print(f'Scanning {source_waveforms_path}')

    invalid_file_extensions = {
        # Metadata / headers
        ".sta", ".hdr", ".inf", ".log", ".meta", ".xml", ".json", ".yml",
        # Event catalogs / parameters
        ".cat", ".hyp", ".pha", ".sum",
        # Response / instrument files
        ".resp", ".pz", ".cal",
        # Processing / analysis outputs
        ".fft", ".psa", ".rsp", ".spec", ".rdm", ".rdt",
        ".mat", ".h5", ".npz",
        # Generic office/text/plots
        ".csv", ".xls", ".xlsx", ".doc", ".docx",
        ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
        # Compression / archives
        ".zip", ".gz", ".tgz", ".bz2", ".xz", ".tar", ".7z",
        # Hidden macOS / Unix files
        ".DS_Store", ".AppleDouble", ".Spotlight-V100",
    }

    files = set()
    for dirpath, dirnames, filenames in os.walk(source_waveforms_path):
        for f in filenames:
            if splitext(f)[1].lower() in invalid_file_extensions:
                continue
        candidate = abspath(join(dirpath, f))
        files.add(candidate)

    pbar = tqdm(
        total=max_rows,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "(estimated remaining time {remaining}s)"
    )

    print(f'Processing records in {source_metadata_path}')
    rec_num = 0
    csv_args = dict(source_metadata_csv_args)
    csv_args.setdefault('chunksize', 100000)
    errs = 0
    write_header = False
    for metadata_chunk in pd.read_csv(source_metadata_path, **csv_args):
        new_metadata = []
        for record in metadata_chunk.itertuples(index=False):

            rec_num += 1
            metadata_row = f'metadata row #{rec_num}'
            record = record._asdict()  # noqa
            pbar.update(1)

            # check returned metadata
            step_name = "_cast_dtype"
            components = {'h1': None, 'h2': None, 'v': None}
            try:
                for f in metadata_fields.keys():
                    step_name = f"_cast_dtype (metadata field '{f}')"
                    dtype = metadata_fields[f]['dtype']
                    record[f] = _cast_dtype(record.get(f), dtype)
                new_metadata.append(record)

                step_name = "find_waveform_paths"
                h1, h2, v = find_waveforms_path(record, files)

                # read waveforms separately:
                for comp_name in components:
                    step_name = f"read_waveform ({comp_name})"
                    comp_path = {'h1': h1, 'h2': h2, 'v': v}[comp_name]
                    if comp_path:
                        components[comp_name] = read_waveform(comp_path, record)

                # save waveforms
                step_name = "save_waveforms"  # noqa
                save_waveforms(dest_waveforms_path, record,
                               components['h1'],
                               components['h2'],
                               components['v'])
            except Exception as exc:
                logging.error(
                    f"Error in {step_name}, {metadata_row}: {exc}"
                )
                errs += 1
                continue

            # if any waveform is None, something went wring, continue but add an error
            no_time_series_num = sum(not _ for _ in components.values())
            if no_time_series_num > 0:
                logging.warning(
                    f"{metadata_row}: {no_time_series_num} of 3 components not created "
                    f"and saved"
                )
                errs += 1

            if rec_num > 100 and errs / rec_num > re_err_th:
                msg = 'Too many errors, check log file and re-run module'
                print(msg, file=sys.stderr)
                logging.error(msg)
                sys.exit(1)

        # save metadata:
        pd.DataFrame(new_metadata).to_csv(
            dest_metadata_path,
            date_format="%Y-%m-%dT%H:%M:%S",
            index=False,
            mode='a',
            header=write_header,
            na_rep=''
        )
        write_header = False

    os.chmod(
        dest_metadata_path,
        os.stat(dest_metadata_path).st_mode | stat.S_IRGRP | stat.S_IROTH
    )
    pbar.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
