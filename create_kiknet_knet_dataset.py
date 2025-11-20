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

import shutil
import zipfile
from typing import Optional, Any, Union, Sequence
import logging
import urllib.request
import os
import time
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
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import glob
# from obspy import read
from io import BytesIO
import math
from tqdm import tqdm
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Waveform:
    """Simple class handling a Waveform (Time History single component)"""
    dt: float
    data: Sequence[float]


########################################################################
# Editable file part: please read carefully and implement your routine #
########################################################################


# The program will stop if the successfully processed waveform ratio falls below this
# value that must be in [0, 1] (this makes spotting errors and checking log faster):
min_waveforms_ok_ratio = 1/5

# max discrepancy between PGA from catalog and computed PGA. A waveform is saved if:
# | PGA - PGA_computed | <= pga_retol * | PGA |
pga_retol = 1/4

# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {
    # 'header': None,  # for CSVs with no header
    'dtype': {  # specifically set dtype (see keys of `source_metadata_fields`)
        'EQ_Code': str,
        "StationCode": str
    }
}
# Mapping from source metadata columns to their new names. Map to None to skip renaming
# and just load the column data
source_metadata_fields = {
    'EQ_Code': 'event_id',
    "StationCode": 'station_id',

    'Origin_Meta': 'origin_time',
    'new_record_start_UTC': 'start_time',
    # 'RecordTime': None,
    'tP_JMA': 'p_wave_arrival_time',
    'tS_JMA': 's_wave_arrival_time',
    'PGA_EW': None,
    'PGA_NS': None,
    'PGA_rotd50': 'PGA',
    # 'azimuth': metadata.get(54),
    'Repi': 'epicentral_distance',
    'Rhypo': 'hypocentral_distance',
    'RJB_0': None,
    'RJB_1': None,
    'Rrup_0': None,
    'Rrup_1': None,
    # 'fault_normal_distance': None,
    "evLat._Meta": 'event_latitude',
    "evLong._Meta": 'event_longitude',
    "Depth. (km)_Meta": 'event_depth',
    "Mag._Meta": 'magnitude',
    "JMA_Magtype": 'magnitude_type',
    # 'depth_to_top_of_fault_rupture': None
    # 'fault_rupture_width': None,
    "fnet_Strike_0": 'strike',
    "fnet_Dip_0": 'dip',
    "fnet_Rake_0": 'rake',
    "fnet_Strike_1": 'strike2',
    "fnet_Dip_1": 'dip2',
    "fnet_Rake_1": 'rake2',
    "Focal_mechanism_BA": 'fault_type',
    "vs30": "vs30",
    "vs30measured": "vs30measured",
    'StationLat.': "station_latitude",
    'StationLong.': "station_longitude",
    'StationHeight(m)': "station_height",
    "z1": "z1",
    "z2pt5": "z2pt5",

    "fc0": "lower_cutoff_frequency_h1",  # FIXME CHECK THIS  hp_h1
    # "fc0": "lower_cutoff_frequency_h2",
    "fc1": "upper_cutoff_frequency_h1",
    # "fc1": "upper_cutoff_frequency_h2",
    # "fc0": "lowest_usable_frequency_h1",
    # "fc1": "lowest_usable_frequency_h2",  # if not sure, leave None
}


def accept_file(file_path) -> bool:
    """Tell whether the given source file can be accepted as time history file"""
    return splitext(file_path)[1] in {
        '.UD1', '.NS1', '.EW1', '.UD2', '.NS2', '.EW2', '.UD', '.NS', '.EW'
    }  # with *1 => borehole


def find_sources(file_path: str, metadata: pd.DataFrame) \
        -> tuple[Optional[str], Optional[str], Optional[str], Optional[pd.Series]]:
    """Find the file paths of the three waveform components, and their metadata

    :param file_path: the waveform path currently processed. Most likely, this is one of
        the three returned waveform paths, adn the other two are inferred from it
    :param metadata: the Metadata dataframe. The returned waveforms metadata must be one
        row of this object as pandas Series, any other object will raise
    """
    root, ext = splitext(file_path)
    if ext == '.EW':  # knet
        paths = file_path, root + ".NS", root + '.UD'
    elif ext == '.NS':
        paths = root + '.EW', file_path, root + '.UD'
    elif ext == '.UD':
        paths = root + '.EW', root + '.NS', file_path
    elif ext == '.EW1':
        paths = file_path, root + ".NS1", root + '.UD1'
    elif ext == '.NS1':
        paths = root + '.EW1', file_path, root + '.UD1'
    elif ext == '.UD1':
        paths = root + '.EW1', root + '.NS1', file_path
    elif ext == '.EW2':
        paths = file_path, root + ".NS2", root + '.UD2'
    elif ext == '.NS2':
        paths = root + '.EW2', file_path, root + '.UD2'
    elif ext == '.UD2':
        paths = root + '.EW2', root + '.NS2', file_path
    else:
        return None, None, None, None

    if not isinstance(metadata.index, pd.MultiIndex) or \
            metadata.index.names != ["event_id", "station_id"]:
        metadata.set_index(["event_id", "station_id"], drop=False, inplace=True)

    record: Optional[pd.Series] = None
    ev_id = basename(dirname(file_path))
    for e in [ev_id, ev_id[2:], ev_id[:-2], ev_id[2:-2]]:
        if root.endswith(e + 'p'):
            sta_id = basename(root)[:6]  # station name is first 6 letters
            try:
                record = metadata.loc[(ev_id, sta_id)].copy()
                if not isinstance(record, pd.Series):  # safety check
                    raise KeyError()
                sta_suffix = f'_{ext[2:3]}' if ext[2:3] else ''
                record["station_id"] += f'{sta_id}{sta_suffix}'
            except KeyError:
                continue
            break
    return paths + (record, )


# def find_waveforms_path(
#         metadata: dict,
#         waveform_file_paths: dict[str, Union[dict, str]]
# ) -> tuple[Optional[str], Optional[str], Optional[str]]:
#     """Get the full source paths of the waveforms (h1, h2 and v components, in this
#     order) from the given row of your source metadata.
#     Paths can be empty or None, meaning that the relative file is missing. This has to
#     be taken into account in `process_waveforms` in case (see below). If files are not
#     missing, then the file must exist
#
#     :param metadata: Python dict, corresponding to a row of your source metadata table.
#         Each dict key represents a Metadata Field (Column). Note that float, str,
#         datetime and categorical values can also be None (e.g., if the Metadata cell
#         was empty)
#     :param waveform_file_paths: dict of string denoting the scanned time histories
#         source directory. It is a dict of strings denoting directory names or file names.
#         If directory names, the mapped value is a nested dict with the same structure,
#         otherwise it is the file absolute path
#     """
#     file_h1 = []
#     file_h2 = []
#     file_v = []
#
#     st_id = metadata['StationCode']
#     orig_time = metadata['RecordTime']
#     # Format as YYMMDDHHMMSS
#     if isinstance(orig_time, str):
#         orig_time = datetime.fromisoformat(orig_time)
#     orig_time_str = orig_time.strftime("%y%m%d%H%M")
#     name = f'{st_id}{orig_time_str}'
#
#     eq_code = str(metadata['EQ_Code'])
#
#     for dir_name in ['knet', 'kik']:
#         dir_name = waveform_file_paths[dir_name]
#         subdir_name1 = eq_code[:4]
#         subdir_name2 = str(int(eq_code[4:6]))
#         try:
#             dir_name = dir_name[subdir_name1][subdir_name2][eq_code]
#         except KeyError:
#             continue
#
#         for file_abs_path in dir_name.values():  # noqa
#             bname = basename(file_abs_path)
#             if not bname.startswith(name):
#                 continue
#             ext = splitext(bname)[1]
#             if ext in ('.EW', '.EW2'):  # see note above
#                 file_h1.append(file_abs_path)
#                 continue
#             if ext in ('.NS', '.NS2'):  # see note above
#                 file_h2.append(file_abs_path)
#                 continue
#             if ext in ('.UD', '.UD2'):  # UD1: borehole
#                 file_v.append(file_abs_path)
#                 continue
#
#     return (
#         file_h1[0] if len(file_h1) == 1 else None,
#         file_h2[0] if len(file_h2) == 1 else None,
#         file_v[0] if len(file_v) == 1 else None
#     )


def read_waveform(full_abs_path: str, content: BytesIO, metadata: pd.Series) -> Waveform:
    """Read a waveform from a file path. Modify according to the format you stored
    your time histories"""
    scale_nom, scale_denom, dt = None, None, None
    for line in content:
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
    rest = content.read()
    data: np.ndarray = np.fromstring(rest, sep=" ", dtype=np.float32)
    # data = np.loadtxt(fp, dtype=np.float32)
    data *= scale_nom / scale_denom / 100.  # the 100. is to convert to m/s**2
    return Waveform(dt, data)


def post_process(
        metadata: pd.Series,
        h1: Optional[Waveform],
        h2: Optional[Waveform],
        v: Optional[Waveform]
) -> tuple[
    pd.Series,
    Optional[Waveform],
    Optional[Waveform],
    Optional[Waveform]
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
    orig_meta, metadata = metadata, metadata.copy()
    metadata['origin_time'] = datetime.fromisoformat(metadata["origin_time"])
    dt_format = "%Y%m%d%H%M%S"
    metadata["start_time"] = \
        datetime.strptime(str(metadata["start_time"]), dt_format)
    metadata["p_wave_arrival_time"] = \
        datetime.strptime(str(metadata["p_wave_arrival_time"]), dt_format)
    metadata["s_wave_arrival_time"] = \
        datetime.strptime(str(metadata["s_wave_arrival_time"]), dt_format)
    metadata["origin_date"] = \
        metadata['origin_time'].replace(hour=0, minute=0, second=0, microsecond=0)
    metadata['fault_type'] = {
        'S': 'Strike-Slip',
        'N': 'Normal',
        'R': 'Reverse'
    }.get(metadata['fault_type'])
    metadata["vs30measured"] = metadata["vs30measured"] in {1, "1", 1.0}
    metadata["region"] = 0
    metadata["filter_type"] = "A"
    metadata["npass"] = 0
    metadata["nroll"] = 0
    metadata["lower_cutoff_frequency_h2"] = metadata["lower_cutoff_frequency_h1"]
    metadata["upper_cutoff_frequency_h2"] = metadata["upper_cutoff_frequency_h1"]
    metadata["lowest_usable_frequency_h1"] = metadata["lower_cutoff_frequency_h1"]
    metadata["lowest_usable_frequency_h2"] = metadata["lower_cutoff_frequency_h2"]
    metadata['magnitude_type'] = {
        'J': 'MJ',  # JMA magnitude
        'D': 'MD',  # JMA displacement magnitude
        'd': 'Md',  # JMA displacement magnitude, but for two stations
        'V': 'MV',  # JMA velocity magnitude
        'v': 'Mv',  # JMA velocity magnitude, but for two or three stations
        'W': 'Mw',  # Moment magnitude
        'B': 'mb',  # Body wave magnitude from USGS
        'S': 'Ms',  # Surface wave magnitude from USGS
    }.get(metadata['magnitude_type'], None)
    return metadata, h1, h2, v



    # origin_time = datetime.fromisoformat(metadata["Origin_Meta"])
    # start_time = datetime.strptime(str(metadata['new_record_start_UTC']),
    #                                "%Y%m%d%H%M%S")
    # p_time = datetime.strptime(str(metadata['tP_JMA']), "%Y%m%d%H%M%S")
    # s_time = datetime.strptime(str(metadata['tS_JMA']), "%Y%m%d%H%M%S")
    # # use datetimes also for event_date (for simplicity when casting later):
    # origin_date = origin_time.replace(hour=0, minute=0, second=0, microsecond=0)
    #
    # # pga check
    # pga1 = metadata['PGA_EW']
    # pga2 = metadata['PGA_NS']
    # pga = metadata['PGA_rotd50']
    #
    # new_metadata = {
    #     'event_id': metadata['EQ_Code'],
    #     # 'azimuth': metadata.get(54),
    #     'epicentral_distance': metadata["Repi"],
    #     'hypocentral_distance': metadata["Rhypo"],
    #     'joyner_boore_distance': pd.Series([metadata["RJB_0"], metadata["RJB_1"]]).mean(),  # noqa
    #     'rupture_distance': pd.Series([metadata["Rrup_0"], metadata["Rrup_1"]]).mean(),
    #     # 'fault_normal_distance': None,
    #     'origin_time': origin_time,
    #     'origin_date': origin_date,
    #     'event_latitude': metadata["evLat._Meta"],
    #     'event_longitude': metadata["evLong._Meta"],
    #     'event_depth': metadata["Depth. (km)_Meta"],
    #     'magnitude': metadata["Mag._Meta"],
    #     'magnitude_type': metadata["JMA_Magtype"],
    #     # 'depth_to_top_of_fault_rupture': None
    #     # 'fault_rupture_width': None,
    #     'strike': metadata["fnet_Strike_0"],
    #     'dip': metadata["fnet_Dip_0"],
    #     'rake': metadata["fnet_Rake_0"],
    #     'strike2': metadata["fnet_Strike_1"],
    #     'dip2': metadata["fnet_Dip_1"],
    #     'rake2': metadata["fnet_Rake_1"],
    #     'fault_type': {
    #         'S': 'Strike-Slip',
    #         'N': 'Normal',
    #         'R': 'Reverse'
    #     }.get(metadata["Focal_mechanism_BA"]),
    #
    #     'station_id': metadata["StationCode"],
    #     "vs30": metadata["vs30"],
    #     "vs30measured": metadata["vs30measured"] in {1, "1", 1.0},
    #     "station_latitude": metadata['StationLat.'],
    #     "station_longitude": metadata['StationLong.'],
    #     "z1": metadata["z1"],
    #     "z2pt5": metadata["z2pt5"],
    #     "region": 0,
    #
    #     # "sensor_type": 'A',
    #     "filter_type": "A",
    #     "npass": 0,
    #     "nroll": 0,
    #     "lower_cutoff_frequency_h1": metadata["fc0"],  # FIXME CHECK THIS  hp_h1
    #     "lower_cutoff_frequency_h2": metadata["fc0"],
    #     "upper_cutoff_frequency_h1": metadata["fc1"],
    #     "upper_cutoff_frequency_h2": metadata["fc1"],
    #     "lowest_usable_frequency_h1": metadata["fc0"],
    #     "lowest_usable_frequency_h2": metadata["fc1"],  # if not sure, leave None
    #     'start_time': start_time,
    #     'p_wave_arrival_time': p_time,
    #     's_wave_arrival_time': s_time,
    #     'PGA': pga
    #
    # }
    #
    # # correct missing values:
    # new_metadata['magnitude_type'] = {
    #     'J': 'MJ',  # JMA magnitude
    #     'D': 'MD',  # JMA displacement magnitude
    #     'd': 'Md',  # JMA displacement magnitude, but for two stations
    #     'V': 'MV',  # JMA velocity magnitude
    #     'v': 'Mv',  # JMA velocity magnitude, but for two or three stations
    #     'W': 'Mw',  # Moment magnitude
    #     'B': 'mb',  # Body wave magnitude from USGS
    #     'S': 'Ms',  # Surface wave magnitude from USGS
    # }.get(new_metadata['magnitude_type'], None)
    #
    # # simply return the arguments (no processing by default):
    # return new_metadata, h1, h2, v


###########################################
# The code below should not be customized #
###########################################


def main():

    try:
        source_metadata_path, source_waveforms_path, dest_root_path = \
            read_script_args(sys.argv)
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    dest_metadata_path = join(dest_root_path, "metadata.hdf")
    dest_waveforms_path = join(dest_root_path, "waveforms")
    print(f"Destination waveforms path: {dest_waveforms_path}")
    print(f"Destination metadata path: {dest_metadata_path}")

    existing = isfile(dest_metadata_path) or isdir(dest_waveforms_path)

    if existing:
        res = input(
            f'Some destination data already exists. Type:\n '
            f'y: delete and re-create all data\n'
            f'm: delete and re-create metadata, save only new waveform files\n'
            f'Any key: quit'
        )
        if res not in ('y', 'm'):
            sys.exit(1)
        if res == 'y' and isdir(dest_waveforms_path):
            shutil.rmtree(dest_waveforms_path)

    if isfile(dest_metadata_path):
        os.unlink(dest_metadata_path)

    if not isdir(dest_root_path):
        os.makedirs(dest_root_path)

    dest_log_path = join(dest_root_path, basename(__file__) + ".log")
    setup_logging(dest_log_path)

    logging.info(f'Working directory: {abspath(os.getcwd())}')
    logging.info(f'Run command      : {" ".join([sys.executable] + sys.argv)}')
    print(f"Source waveforms path: {source_waveforms_path}")
    print(f"Source metadata path:  {source_metadata_path}")

    # Reading metadata fields dtypes and info:
    try:
        dest_metadata_fields_path = join(dest_root_path, 'metadata_fields.yml')
        metadata_fields = get_metadata_fields(dest_metadata_fields_path)
        with open(dest_metadata_fields_path, "r") as _:
            logging.info(f'Metadata fields file: {dest_metadata_fields_path}')
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f'Scanning source waveforms directory...', end=" ", flush=True)
    files = scan_dir(source_waveforms_path)
    print(f'{len(files):,} file(s) found')

    pbar = tqdm(
        total=len(files),
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "(estimated remaining time {remaining}s)"
    )

    print(f'Reading source metadata file...', end=" ", flush=True)
    csv_args = dict(source_metadata_csv_args)
    # csv_args.setdefault('chunksize', 10000)
    csv_args.setdefault('usecols', source_metadata_fields.keys())
    metadata = pd.read_csv(source_metadata_path, **csv_args)
    metadata = metadata.rename(
        columns={k: v for k, v in source_metadata_fields.items() if v is not None}    # RHB1 and SITE_CLASSIFICATION_EC8  # FIXME DO!
    )
    for lbl in ['event_id', 'station_id']:
        metadata[lbl] = metadata[lbl].astype('category')
        metadata_fields[lbl]['dtype'] = metadata[lbl].dtype
    print(f'{len(metadata):,} record(s), {len(metadata.columns):,} field(s) per record')

    print(f'Creating harmonized dataset from source')
    records = []
    item_num = 0
    errs = 0
    while len(files):
        num_files = 1
        file = files.pop()
        step_name = ""

        try:
            step_name = "find_related"
            h1_path, h2_path, v_path, record = find_sources(file, metadata)

            # checks:
            if sum(_ is not None for _ in (h1_path, h2_path, v_path)) == 0:
                raise Exception('No existing file found')
            if not isinstance(record, pd.Series):
                if isinstance(record, pd.DataFrame):
                    raise Exception('Multiple metadata record found')
                raise Exception('No metadata record found')
            for _ in (h1_path, h2_path, v_path):
                if _ in files:
                    num_files += 1
                    files.remove(_)

            comps = {}
            for cmp_name, cmp_path in zip(('h1', 'h2', 'v'), (h1_path, h2_path, v_path)):
                step_name = f"read_waveform ({cmp_name})"
                if cmp_path:
                    with open_file(cmp_path) as file_p:
                        comps[cmp_name] = read_waveform(cmp_path, file_p, record)

            if all(_ is None for _ in comps.values()):
                raise Exception('No waveform read')
            if len(set(_.dt for _ in comps.values())) != 1:
                raise Exception('Waveform components have mismatching dt')

            # process waveforms
            step_name = "save_waveforms"  # noqa
            h1, h2, v = comps.get('h1'), comps.get('h2'), comps.get('v')
            # old_record = dict(record)  # for testing purposes
            new_record, h1, h2, v = post_process(record, h1, h2, v)

            # check record data types:
            item_num += 1
            clean_record = {'id': item_num}
            for f in new_record.keys():
                if f not in metadata_fields:
                    continue
                default_val = metadata_fields[f].get('default')
                val = new_record.get(f, default_val)
                step_name = f"_cast_dtype (field '{f}')"
                dtype = metadata_fields[f]['dtype']
                try:
                    clean_record[f] = cast_dtype(val, dtype)
                except AssertionError:
                    raise AssertionError(f'Invalid value for "{f}": {str(val)}')

            # final checks:
            check_final_metadata(clean_record, h1, h2)

            # finalize clean_record:
            avail_comps, sampling_rate = extract_metadata_from_waveforms(h1, h2, v)
            clean_record['available_components'] = avail_comps
            clean_record['sampling_rate'] = sampling_rate if \
                pd.notna(sampling_rate) else \
                int(metadata_fields['sampling_rate']['default'])

            # save metadata:
            step_name = "save_metadata"  # noqa
            records.append(clean_record)
            if len(records) > 1000:
                save_metadata(dest_metadata_path, pd.DataFrame(records), metadata_fields)
                records = []

            # save waveforms
            step_name = "save_waveforms"  # noqa
            file_path = join(dest_waveforms_path, get_file_path(clean_record))
            if not isfile(file_path):
                save_waveforms(file_path, h1, h2, v)

        except Exception as exc:
            logging.error(
                f"[ERROR] {file}: {exc} | Step: `{step_name}`"
            )
            errs += 1
        finally:
            pbar.update(num_files)

        if pbar.n / pbar.total > (1 - min_waveforms_ok_ratio):
            # we processed enough data (1 - waveforms_ok_ratio)
            ok_ratio = 1 - (errs / pbar.total)
            if ok_ratio < min_waveforms_ok_ratio:
                # the processed data error ratio is too high:
                msg = f'Too many errors ({errs} of {pbar.total} records)'
                print(msg, file=sys.stderr)
                logging.error(msg)
                sys.exit(1)

    if isfile(dest_metadata_path):
        if len(records):
            save_metadata(dest_metadata_path, pd.DataFrame(records), metadata_fields)
        os.chmod(
            dest_metadata_path,
            os.stat(dest_metadata_path).st_mode | stat.S_IRGRP | stat.S_IROTH
        )

    pbar.close()
    sys.exit(0)


def read_script_args(sys_argv):

    if len(sys_argv) != 2:
        raise ValueError(f'Error: invalid argument, provide a valid yaml file')

    yaml_path = sys_argv[1]
    if not isfile(yaml_path):
        raise ValueError(f'Error: the file {yaml_path} does not exist')

    try:
        with open(yaml_path) as _:
            data = yaml.safe_load(_)
        return data['source_metadata'], data['source_data'], data['destination']

    except Exception as exc:
        raise ValueError(f'Error: {yaml_path} error: {exc}')


def setup_logging(filename):
    logger = logging.getLogger()  # root logger
    # if not logger.handlers:
    handler = logging.FileHandler(filename, mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def scan_dir(source_root_dir) -> set[str]:
    """Scan the given directory. Zip files are opened and treated as directories
    Use open_file to open any returned file path
    """
    files = set()
    for entry in os.scandir(source_root_dir):
        if entry.is_dir():
            files.update(scan_dir(entry.path))
            continue
        file_path = abspath(entry.path)
        if accept_file(file_path):
            files.add(file_path)
        elif splitext(entry.name)[1].lower() == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                for name in z.namelist():
                    file_path2 = join(file_path, name)
                    if accept_file(file_path2):
                        files.add(file_path2)
    return files


def open_file(file_path) -> BytesIO:
    """
    Open a regular file or a file inside a .zip archive. file_path is any item
    returned by `scan_dir`
    """
    fp_lower = file_path.lower()
    if f".zip{os.sep}" in fp_lower:
        idx = fp_lower.index(".zip")
        zip_path, inner_path = file_path[:idx + 4], file_path[idx + 5:]
        with zipfile.ZipFile(zip_path, "r") as z:
            data = z.read(inner_path)
    elif fp_lower.endswith(".zip"):
        zip_path = file_path
        with zipfile.ZipFile(zip_path, "r") as z:
            namelist = z.namelist()
            if len(namelist) != 1:
                raise ValueError(
                    f"{file_path} contains {len(namelist)} files, expected one only")
            data = z.read(namelist[0])
    else:
        with open(file_path, "rb") as f:
            data = f.read()

    return BytesIO(data)


def get_metadata_fields(dest_path):
    """
    Get the YAML metadat fields and saves it
    into dest_root_path. Returns the dict of the parsed yaml
    """
    with open(join(dirname(__file__), 'metadata_fields.yml'), 'rb') as _:
        metadata_fields_content = _.read()
        # Load YAML into Python dict
        metadata_fields = yaml.safe_load(metadata_fields_content.decode("utf-8"))
        # save to file
        with open(dest_path, "wb") as f:
            f.write(metadata_fields_content)
        # convert dtypes:
        for m in metadata_fields:
            field = metadata_fields[m]
            if isinstance(field['dtype'], (list, tuple)):
                assert 'default' not in field or field['default'] in m['dtype']
                field['dtype'] = pd.CategoricalDtype(field['dtype'])

    return metadata_fields


def cast_dtype(val: Any, dtype: Union[str, pd.CategoricalDtype]):
    if dtype == 'int':
        assert isinstance(val, int) or (isinstance(val, float) and int(val) == val)
        val = int(val)
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
        elif isinstance(dtype, pd.CategoricalDtype):
            assert val in dtype.categories
    return val


def get_file_path(metadata: dict):
    """Return the file (relative) path from the given metadata
    (record metadata already cleaned)"""
    return join(
        str(metadata['event_id']),
        str(metadata['station_id']) + ".h5"
    )


def check_final_metadata(metadata: dict, h1: Optional[Waveform], h2: Optional[Waveform]):

    pga = metadata['PGA']
    pga_c = None
    if h1 is not None and h2 is not None:
        pga_c = np.sqrt(np.max(np.abs(h1.data)) * np.max(np.abs(h2.data)))
    elif h1 is not None and h2 is None:
        pga_c = np.max(np.abs(h1.data))
    elif h1 is None and h2 is not None:
        pga_c = np.max(np.abs(h2.data))
    if pga_c is not None:
        rtol = pga_retol
        assert np.isclose(pga_c, pga, rtol=rtol, atol=0), \
            f"|PGA - PGA_computed| > {rtol} * | PGA |"

    o_time = 'origin_time' if pd.notna(metadata['origin_time']) else 'origin_date'
    assert pd.notna(metadata[o_time]), f"{o_time} is NA"

    for t_before, t_after in [
        (o_time, 'p_wave_arrival_time'),
        ('p_wave_arrival_time', 's_wave_arrival_time')
    ]:
        val_before = metadata.get(t_before)
        val_after = metadata.get(t_after)
        if pd.notna(val_before) and pd.notna(val_after):
            assert val_after > val_before, f"{t_after} must happen after {t_before}"


def extract_metadata_from_waveforms(
        h1: Optional[Waveform],
        h2: Optional[Waveform],
        v: Optional[Waveform]
) -> tuple[Optional[str], Optional[int]]:
    dt = None
    avail_comps = ''
    for comp, avail_comp_str in zip((h1, h2, v), ('H', 'H', 'V')):
        if comp is None:
            continue
        if avail_comps == '':  # first non null component
            dt = comp.dt
        elif comp.dt != dt:
            dt = None
        avail_comps += avail_comp_str

    sampling_rate = int(1./dt) if dt is not None and int(1./dt) == 1./dt else None
    return avail_comps, sampling_rate


def save_metadata(dest_metadata_path: str, metadata: pd.DataFrame, metadata_fields):
    if metadata is not None and not metadata.empty:
        # save metadata:
        new_metadata_df = pd.DataFrame(metadata)
        new_metadata_df = new_metadata_df[
            [c for c in new_metadata_df if c in metadata_fields]
        ].copy()
        for col in metadata_fields:
            new_metadata_df[col] = cast_dtypes(
                metadata_fields[col]['dtype'],
                metadata_fields[col].get('default'),
                new_metadata_df.get(col),
                new_metadata_df
            )
        hdf_kwargs = {
            'key': "metadata",  # table name
            'mode': "a",
            # (1st time creates a new file because we deleted it, see above)
            'format': "table",  # required for appendable table
            'append': True,  # first batch still uses append=True
            # 'min_itemsize': {
            #     'event_id': metadata["event_id"].str.len,
            #     'station_id': metadata["station_id"].str.len,
            # },  # required for strings (used only the 1st time to_hdf is called)  # noqa
            # 'data_columns': [],  # list of columns you need to query these later
        }
        new_metadata_df.to_hdf(dest_metadata_path, **hdf_kwargs)


def cast_dtypes(
        dtype: Union[str, pd.CategoricalDtype],
        default_value: Any,
        dataframe_column: str,
        dataframe: pd.DataFrame
):
    """
    Cast the values of the output metadata. Safety method (each value in `values` is
    the outcome of `cast_dtype` so it should be of the correct dtype already)
    """
    values = dataframe.get(dataframe_column)
    if dtype == 'int':
        if values is None:
            values = [default_value] * len(dataframe)
        # assert pd.notna(values).all()
        return values.astype(int)
    elif dtype == 'bool':
        if values is None:
            values = [default_value] * len(dataframe)
        return values.astype(bool)
    elif dtype == 'datetime':
        if values is None:
            if pd.isna(default_value):
                default_value = pd.NaT
            values = [default_value] * len(dataframe)
        return pd.to_datetime(values, errors='coerce')
    elif dtype == 'str':
        if values is None:
            if pd.isna(default_value):
                default_value = None
            values = [default_value] * len(dataframe)
        return values.astype(str)
    elif dtype == 'float':
        if values is None:
            if pd.isna(default_value):
                default_value = np.nan
            values = [default_value] * len(dataframe)
        return values.astype(float)
    elif isinstance(dtype, pd.CategoricalDtype):
        if values is None:
            if pd.isna(default_value):
                default_value = None
            values = [default_value] * len(dataframe)
        else:
            cat_values = set(dtype.categories)  # allowed categories
            invalid = set(values.dropna()) - cat_values  # invalid, non-NA values
            if invalid:
                raise AssertionError(
                    f'Unrecognized categories in {dataframe_column}: {invalid}'
                )
        return values.astype(dtype)
    raise AssertionError(f'Unrecognized dtype {dtype}')


def save_waveforms(
        file_path, h1: Optional[Waveform], h2: Optional[Waveform], v: Optional[Waveform]
):
    if not isdir(dirname(file_path)):
        os.makedirs(dirname(file_path))

    dts = {x.dt for x in (h1, h2, v) if x is not None}
    assert len(dts), "No waveform to save"  # safety check
    data_has_samples = {len(x.data) for x in (h1, h2, v) if x is not None}
    assert all(data_has_samples), "Cannot save empty waveform(s)"

    assert len(dts) == 1, "Non-unique dt in waveforms"
    dt = dts.pop() if dts else None

    empty = np.array([])
    with h5py.File(file_path, "w") as f:
        # Save existing components
        f.create_dataset("h1", data=empty if h1 is None else h1.data)
        f.create_dataset("h2", data=empty if h2 is None else h2.data)
        f.create_dataset("v", data=empty if v is None else v.data)
        f.attrs["dt"] = dt

    # Add read permission for group (stat.S_IRGRP) and others (stat.S_IROTH).
    if isfile(file_path):
        os.chmod(file_path, os.stat(file_path).st_mode | stat.S_IRGRP | stat.S_IROTH)


def read_wafevorms(file_path) -> (float, np.ndarray, np.ndarray, np.ndarray):
    """Reads the file path previously saved. NOT USED, HERE ONLY FOR REF"""
    with h5py.File(file_path, "r") as f:
        dt = f.attrs["dt"]
        h1 = f['h1'][:]
        h2 = f['h2'][:]
        v = f['v'][:]
    return dt, h1, h2, v


if __name__ == "__main__":
    main()
