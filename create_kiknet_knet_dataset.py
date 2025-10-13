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
# from obspy import read
from io import BytesIO
import math
from tqdm import tqdm


########################################################################
# Editable file part: please read carefully and implement your routine #
########################################################################

def accept_file(file_path) -> bool:
    """Tell whether the given source file can be accepted as time history file"""
    return 'HYG012' in file_path and splitext(file_path)[1] in {'.UD2', '.NS2', '.EW2', '.UD', '.NS', '.EW'}  # with *1 => borehole  # FIXME


def find_waveforms_path(metadata: dict, waveform_file_paths: set[str]) \
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
    file_h1 = []
    file_h2 = []
    file_v = []

    st_id = metadata['StationCode']
    orig_time = metadata['RecordTime']
    # Format as YYMMDDHHMMSS
    if isinstance(orig_time, str):
        orig_time = datetime.fromisoformat(orig_time)
    orig_time_str = orig_time.strftime("%y%m%d%H%M")
    name = f'{st_id}{orig_time_str}'

    for file_abs_path in waveform_file_paths:
        bname = basename(file_abs_path)
        dir_bname = basename(dirname(file_abs_path))
        if dir_bname != str(metadata['EQ_Code']) or not bname.startswith(name):
            continue
        ext = splitext(bname)[1]
        if ext in ('.UD', '.UD2'):  # UD1: borehole
            file_v.append(file_abs_path)
            continue
        if ext in ('.NS', '.NS2'):  # see note above
            file_h2.append(file_abs_path)
            continue
        if ext in ('.EW', '.EW2'):  # see note above
            file_h1.append(file_abs_path)
            continue

    return (
        file_h1[0] if len(file_h1) == 1 else None,
        file_h2[0] if len(file_h2) == 1 else None,
        file_v[0] if len(file_v) == 1 else None
    )


def read_waveform(full_abs_path: str, metadata: dict) -> tuple[float, ndarray]:
    """Read a waveform from a file path. Modify according to the format you stored
    your time histories"""
    scale_nom, scale_denom, dt = None, None, None
    with open(full_abs_path, "rb") as fp:
        for line in fp:
            if line.startswith(b'Sampling Freq(Hz)'):
                dt = 1.0 / float(
                    line.strip().lower().split(b' ')[-1].removesuffix(b'hz'))
            elif line.startswith(b'Scale Factor'):
                scale_str = line.split(b'  ', 1)[1].strip().split(b'/')
                scale_nom = float(scale_str[0][:-5])
                scale_denom = float(scale_str[1])
            elif line.startswith(b'Memo'):
                if any(_ is None for _ in (scale_nom, scale_denom, dt)):
                    raise ValueError('dt /scale nom / scale denom not found')
                break
        rest = fp.read()
        data = np.fromstring(rest, sep=" ", dtype=np.float32)
        # data = np.loadtxt(fp, dtype=np.float32)
        data *= scale_nom / scale_denom / 100.  # the 100. is to convert to m/s**2
    return dt, data


def process_waveforms(
        metadata: dict,
        h1: Optional[tuple[float, ndarray]],
        h2: Optional[tuple[float, ndarray]],
        v: Optional[tuple[float, ndarray]]
) -> tuple[
    dict,
    Optional[tuple[float, ndarray]],
    Optional[tuple[float, ndarray]],
    Optional[tuple[float, ndarray]]
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
    origin_time = datetime.fromisoformat(metadata["Origin Meta"]).isoformat()
    start_time = datetime.strptime(metadata['new_record_start_UTC'],
                                   "%Y%m%d%H%M%S").isoformat()
    pga1 = metadata['PGA_EW']
    pga2 = metadata['PGA_NS']

    new_metadata = {
        'event_id': metadata['EQ_Code'],
        # 'azimuth': metadata.get(54),
        'epicentral_distance': metadata["Repi"],
        'hypocentral_distance': metadata["Rhypo"],
        'joyner_boore_distance': pd.Series([metadata["RJB_0"], metadata["RJB_1"]]).mean(),  # noqa
        'rupture_distance': pd.Series([metadata["Rrup 0"], metadata["Rrup 1"]]).mean(),
        # 'fault_normal_distance': None,
        'origin_time': origin_time,
        'event_lat': metadata["evLat. Meta"],
        'event_lon': metadata["evLong. Meta"],
        'event_depth': metadata["Depth. (km) Meta"],
        'magnitude': metadata["Earthquake Magnitude"],
        'magnitude_type': metadata["JMA_Magtype"],
        # 'depth_to_top_of_fault_rupture': None
        # 'fault_rupture_width': None,
        'strike': metadata["fnet_Strike_0"],
        'dip': metadata["fnet_Dip_0"],
        'rake': metadata["fnet_Rake_0"],
        'strike2': metadata["fnet_Strike_1"],
        'dip2': metadata["fnet_Dip_1"],
        'rake2': metadata["fnet_Rake_1"],
        'fault_type': {'S': 0 ,'N' :1, 'R' : 2}.get(metadata["Focal_mechanism_BA"]),

        'station_id': metadata["StationCode"],
        "vs30": metadata["vs30"],
        "vs30measured": metadata["vs30measured"] in {1, "1", 1.0},
        "station_lat": metadata['StationLat.'],
        "station_lon": metadata['StationLong.'],
        "z1": metadata["z1pt0"],
        "z2pt5": metadata["z2pt5"],
        "region": 0,

        # "sensor_type": 'A',
        "filter_type": "A",
        "npass": 4,  # 4th-order acausal butterworth filter,  # FIXME see doc dino
        "nroll": 0,  # metadata["nroll"],  # FIXME see doc dino
        "lower_cutoff_frequency_h1": metadata["fc0"],  # FIXME CHECK THIS  hp_h1
        "lower_cutoff_frequency_h2": metadata["fc0"],
        "upper_cutoff_frequency_h1": metadata["fc1"],
        "upper_cutoff_frequency_h2": metadata["fc1"],
        "lowest_usable_frequency_h1": metadata["fc0"],
        "lowest_usable_frequency_h2": metadata["fc1"],  # if not sure, leave None
        'start_time': start_time,
        'p_wave_arrival_time': metadata['tP_JMA'],
        's_wave_arrival_time': metadata['tS_JMA'],
        'PGA': math.sqrt(pga1*pga2) if pd.notna([pga1, pga2]).all() else None

    }

    # correct missing values:
    if new_metadata['mag_type'] == 'U':
        new_metadata['mag_type'] = None

    # simply return the arguments (no processing by default):
    return new_metadata, h1, h2, v


# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {}  # {'header': None} for CSVs with no header

# relative error threshold. After 100 waveforms, when waveforms with error / warnings
# get higher than this number (relative to the total number of processed waveforms)
# the program will stop. 0.05 means 5% max of erroneous waveforms
re_err_th = 0.05


###########################################
# The code below should not be customized #
###########################################


def save_waveforms(
        file_path,
        h1: Optional[tuple[float, ndarray]],
        h2: Optional[tuple[float, ndarray]],
        v: Optional[tuple[float, ndarray]]
):
    if not isdir(dirname(file_path)):
        os.makedirs(dirname(file_path))

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
    if isfile(file_path):
        os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IRGRP | stat.S_IROTH)


def _cast_dtype(val: Any, dtype: str):
    if dtype == 'int':
        assert isinstance(val, int) or (isinstance(val, float) and int(val) == val)
        return int(val)
    elif dtype == 'bool':
        if val in {0, 1}:
            val = bool(val)
        assert isinstance(val, bool)
    elif val is not None:
        if dtype == 'datetime':
            if hasattr(val, 'to_pydatetime'):  # for safety
                val = val.to_pydatetime()
            assert isinstance(val, datetime)
        elif val == 'str':
            assert isinstance(val, str)
        elif val == 'float':
            assert isinstance(val, float)
        elif isinstance(dtype, (list, tuple)):
            assert val in dtype
    return val


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
    err_no_file = False
    if not err_noarg:
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

    dest_root_path = get_dest_dir_path()
    dest_metadata_path = join(dest_root_path, splitext(basename(__file__))[0] + ".csv")
    dest_waveforms_path = join(dest_root_path, "waveforms")

    if not isdir(dest_root_path):
        os.makedirs(dest_root_path)

    log_dest_path = dest_metadata_path + ".log"
    setup_logging(log_dest_path)

    logging.info(f'Script: {" ".join([sys.executable] + sys.argv)}')
    print(f"Source metadata path: {source_metadata_path}")
    print(f"Source waveforms path: {source_waveforms_path}")

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

    print(f'Scanning {source_waveforms_path}')

    files = set()
    for dirpath, dirnames, filenames in os.walk(source_waveforms_path):
        for f in filenames:
            candidate = abspath(join(dirpath, f))
            if accept_file(candidate):
                files.add(candidate)

    print(f'Found {len(files)} file(s) as time history candidates')
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
    write_header = True
    for metadata_chunk in pd.read_csv(source_metadata_path, **csv_args):
        new_metadata = []
        for record in metadata_chunk.to_dict(orient="records"):

            rec_num += 1
            metadata_row = f'metadata row #{rec_num}'
            pbar.update(1)

            step_name = ""
            components = {'h1': None, 'h2': None, 'v': None}
            try:
                step_name = "find_waveform_paths"
                h1, h2, v = find_waveforms_path(record, files)

                # read waveforms separately:
                f_name = h1 if h1 else h2 if h2 else v if v else None

                if f_name is not None:
                    for comp_name in components:
                        step_name = f"read_waveform ({comp_name})"
                        comp_path = {'h1': h1, 'h2': h2, 'v': v}[comp_name]
                        if comp_path:
                            components[comp_name] = read_waveform(comp_path, record)

                    # process waveforms
                    step_name = "save_waveforms"  # noqa
                    old_record = dict(record)  # for testing purposes
                    new_record, h1, h2, v = process_waveforms(
                        record,
                        components.get('h1'),
                        components.get('h2'),
                        components.get('v')
                    )
                    # check record data types:
                    clean_record = {'id': rec_num}
                    for f in new_record.keys():
                        default_val = metadata_fields[f].get('default')
                        val = new_record.get(f, default_val)
                        step_name = f"_cast_dtype (field '{f}')"
                        dtype = metadata_fields[f]['dtype']
                        try:
                            clean_record[f] = _cast_dtype(val, dtype)
                        except AssertionError:
                            raise AssertionError(f'Field {f}: invalid value {str(val)}')
                    new_metadata.append(clean_record)

                    # save waveforms
                    step_name = "save_waveforms"  # noqa
                    clean_record['file_path'] = join(
                        str(clean_record['event_id']),
                        splitext(basename(f_name))[0] + ".h5"
                    )
                    save_waveforms(join(dest_waveforms_path, clean_record['file_path']),
                        h1, h2, v
                    )
                    avail_th = None
                    if h1 is None and h2 is None and v is not None:
                        avail_th = 'V'
                    elif h1 is not None and h2 is not None and v is not None:
                        avail_th = 'HHV'
                    elif h1 is not None and h2 is not None and v is None:
                        avail_th = 'HH'
                    elif (h1 is None) != (h2 is None) and v is not None:
                        avail_th = 'HV'
                    elif (h1 is not None) != (h2 is not None) and v is None:
                        avail_th = 'H'
                    clean_record['available_components'] = avail_th

            except Exception as exc:
                logging.error(
                    f"Error in {step_name}, {metadata_row}: {exc}"
                )
                errs += 1
                continue

            # if any waveform is None, something went wrong, continue but add an error
            _time_series_num = sum(_ is not None for _ in components.values())
            if _time_series_num < 3:
                logging.warning(
                    f"{metadata_row}: only {_time_series_num} of 3 components created "
                    f"and saved"
                )
                errs += 1

            if rec_num > 100 and errs / rec_num > re_err_th:
                msg = 'Too many errors, check log file and re-run module'
                print(msg, file=sys.stderr)
                logging.error(msg)
                sys.exit(1)

        if new_metadata:
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

    if isfile(dest_metadata_path):
        os.chmod(
            dest_metadata_path,
            os.stat(dest_metadata_path).st_mode | stat.S_IRGRP | stat.S_IROTH
        )
    pbar.close()
    sys.exit(0)


def setup_logging(filename):
    logger = logging.getLogger()  # root logger
    # if not logger.handlers:
    handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


if __name__ == "__main__":
    main()
