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

    rsn = metadata[0]

    for file_abs_path in waveform_file_paths:
        bname = basename(file_abs_path)
        if bname.endswith(f'{rsn}_{metadata[112].strip()}'):
            file_h1.append(file_abs_path)
            continue
        if bname.endswith(f'{rsn}_{metadata[113].strip()}'):
            file_h2.append(file_abs_path)
            continue
        if bname.endswith(f'{rsn}_{metadata[114].strip()}'):
            file_v.append(file_abs_path)
            continue

    return (
        file_h1[0] if len(file_h1) == 1 else None,
        file_h2[0] if len(file_h2) == 1 else None,
        file_v[0] if len(file_v) == 1 else None
    )


def read_waveform(full_abs_path: str, metadata: dict) -> tuple[float, ndarray]:
    """Read a waveform from a file path. Modify according to the format you stored
    your time histories"""
    with open(full_abs_path) as f:
        # First few lines are headers
        header1 = f.readline().strip()
        header2 = f.readline().strip()
        header3 = f.readline().strip()
        header4 = f.readline().split(",")
        npts = int(re.match(r"NPTS\s*=\s*(\d+)", header4[0].strip()).group(1))
        dt = float(re.match(r"DT\s*=\s*([\.\d]+)\s*SEC", header4[1].strip()).group(1))
        data_str = " ".join(line for line in f)
        # The acceleration time series is given in units of g. So I convert it in m/s.
        return dt, np.fromstring(data_str, sep=" ") * 9.80665

        # The acceleration time series is given in units of g. So I convert it in m/s.
        # However it is said only that its in the units of g, not sure if its 980 cm/s^ or 9.8 m/s^2
        # g = 9.8  # in (m/s^2)
        # data = np.array([number * g for number in data])
        # pga = max([abs(number) for number in data])
        # pga = max(np.abs(data))
        # pga_ind = [abs(number) for number in data].index(pga)
        # pga_ind = np.argmax(np.abs(data))
        # return dt, data


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
    # compute arrival time (correct)
    year = metadata[3]
    month_day = str(metadata[4])
    if len(month_day) == 3:
        month_day = '0' + month_day
    hour_min = str(metadata[5])
    if len(hour_min) == 3:
        hour_min = '0' + hour_min
    assert len(month_day) == len(hour_min) == 4, 'month_day or hour_min invalid'
    month, day = int(month_day[:2]), int(month_day[2:])
    hour, min = int(hour_min[:2]), int(hour_min[2:])
    evt_time = datetime(year=year, month=month, day=day, hour=hour, minute=min)
    evt_id = metadata.get(1)
    new_metadata = {
        'evt_id': evt_id,
        # 'azimuth': metadata.get(54),
        'repi': metadata.get(47),
        'rhypo': metadata.get(48),
        'rjb': metadata.get(49),
        'rrup': metadata.get(52),
        # 'r_x': None,
        # 'r_y': None,
        # 'r_volc': None,
        'evt_time': evt_time,
        'evt_lat': metadata.get(24),
        'evt_lon': metadata.get(25),
        'evt_depth': metadata.get(26),
        'mag': metadata.get(9),
        'mag_type': metadata.get(10),

        'rup_top_depth': metadata.get(31),
        'rup_width': metadata.get(33),  # FIXME check
        'strike': metadata.get(16),
        'dip': metadata.get(17),
        'rake': metadata.get(18),
        'fault_type': metadata.get(19),  # FIXME check

        'sta_id': str(metadata.get(8)) if str(metadata.get(8)) != "99999" else
        "NGA#:" + str(metadata.get(7)),
        "backarc": False,  # FIXME check!
        "sta_lat": metadata.get(87),
        "sta_lon": metadata.get(88),
        "z1": metadata.get(97),
        "z2pt5": metadata.get(98),
        "vs30": metadata.get(79),
        "vs30measured": metadata.get(81) in {0, "0", 0.0},

        # "xvf": None,
        "region": 0,
        "fpeak": None,
        # "geology": None,
        "sensor_type": 'A',
        # "fpath": f"{evt_id}/{splitext(basename(metadata.get(116)))[0] + '.h5'}",
        # "fpath_h1": f"{evt_id}/{metadata.get(116)}" if metadata.get() else None,
        # "fpath_h2": f"{evt_id}/{metadata.get(117)}" if metadata.get() else None,
        # "fpath_v": f"{evt_id}/{metadata.get(118)}" if metadata.get() else None,
        "filter_type": metadata.get(114),

        "npass": metadata.get(115),
        "nroll": metadata.get(116),
        "hp_h1": metadata.get(117),
        "hp_h2": metadata.get(118),
        "lp_h1": metadata.get(119),
        "lp_h2": metadata.get(120),
        "luf_h1": metadata.get(122),
        "luf_h2": metadata.get(123)
    }
    # simply return the arguments (no processing by default):
    return new_metadata, h1, h2, v


# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {'header': None}

# relative error threshold. After 100 waveforms, when waveforms with error / warnings
# get higher than this number (relative to the total number of processed waveforms)
# the program will stop. 0.05 means 5% max of erroneous waveforms
re_err_th = 1.01


###########################################
# The code below should not be customized #
###########################################


def save_waveforms(
        file_path,
        h1: Optional[tuple[float, ndarray]],
        h2: Optional[tuple[float, ndarray]],
        v: Optional[tuple[float, ndarray]]
):
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
        assert val is None or isinstance(val, (int, float))
        val = float(val)
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
        ".mat", ".h5", ".npz", ".py", ".pyc", ".java", ".c",
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
                    record, h1, h2, v = process_waveforms(
                        record,
                        components.get('h1'),
                        components.get('h2'),
                        components.get('v')
                    )
                    # check record data types:
                    new_record = {}
                    for f in record.keys():
                        if f == 'fault_type':
                            asd = 9
                        step_name = f"_cast_dtype (field '{f}')"
                        dtype = metadata_fields[f]['dtype']
                        default_val = metadata_fields[f].get('default')
                        try:
                            new_record[f] = _cast_dtype(record.get(f, default_val), dtype)
                        except AssertionError:
                            raise AssertionError(f'Field {f}: invalid value '
                                                 f'{str(record.get(f, default_val))}')
                    new_metadata.append(new_record)

                    # save waveforms
                    step_name = "save_waveforms"  # noqa
                    dest_waveforms_path = join(
                        dest_root_path, splitext(basename(f_name))[0] + ".h5"
                    )
                    save_waveforms(dest_waveforms_path, h1, h2, v)
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
                    new_record['avail_time_hist'] = avail_th

            except Exception as exc:
                logging.error(
                    f"Error in {step_name}, {metadata_row}: {exc}"
                )
                errs += 1
                continue

            # if any waveform is None, something went wring, continue but add an error
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

    os.chmod(
        dest_metadata_path,
        os.stat(dest_metadata_path).st_mode | stat.S_IRGRP | stat.S_IROTH
    )
    pbar.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
