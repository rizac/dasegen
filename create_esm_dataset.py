"""
Python module to be executed as script to generate a new harmonized dataset from
heterogeneous sources. For a new dataset, copy rename this or any look-alike modules in
this directory and modify the editable part (see below). More details in README.md
"""
from __future__ import annotations

import shutil
import zipfile
from typing import Optional, Any, Union, Sequence
import logging
import urllib.request
import warnings
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
from io import BytesIO
import math
from tqdm import tqdm
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Waveform:
    """Simple class handling a Waveform (Time History single component)"""
    dt: float
    data: ndarray[float]


########################################################################
# Editable file part: please read carefully and implement your routine #
########################################################################


# The program will stop if the successfully processed waveform ratio falls below this
# value that must be in [0, 1] (this makes spotting errors and checking log faster):
min_waveforms_ok_ratio = 1/100

# max discrepancy between PGA from catalog and computed PGA. A waveform is saved if:
# | PGA - PGA_computed | <= pga_retol * | PGA |
pga_retol = 1/4

# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {
    'sep': ';'
    # 'header': None  # for CSVs with no header
    # 'dtype': {}  # NOT RECOMMENDED, see `metadata_fields.yml` instead
    # 'usecols': []  # NOT RECOMMENDED, see `source_metadata_fields` below instead
}

# Mapping from source metadata columns to their new names. Map to None to skip renaming
# and just load the column data
source_metadata_fields = {
    'event_id': "event_id",
    "network_code" :None,
    "station_code": None,
    "location_code": None,
    "instrument_code": None,

    "epi_dist": "epicentral_distance",
    # "?": "hypocentral_distance",
    "JB_dist": "joyner_boore_distance",
    "rup_dist": "rupture_distance",
    "Rx_dist": "fault_normal_distance",
    'event_time': "origin_time",

    "ev_latitude": "event_latitude",
    "ev_longitude": "event_longitude",
    "ev_depth_km": "event_depth",
    "EMEC_Mw": "magnitude",
    "Mw": None,
    "ML": None,
    "Ms": None,
    "es_z_top": "depth_to_top_of_fault_rupture",
    "es_width": "fault_rupture_width",
    "es_strike": "strike",
    "es_dip": "dip",
    "es_rake": "rake",

    "fm_type_code": "fault_type",
    "vs30_m_sec": "vs30",
    'vs30_meas_type': None,
    'vs30_m_sec_WA': None,

    # vs30measured is a boolean expression; treated as key
    # "Measured/Inferred Class": "vs30measured",
    "st_latitude": "station_latitude",
    "st_longitude": "station_longitude",
    # "Northern CA/Southern CA - H11 Z1 (m)": "z1",
    # "Northern CA/Southern CA - H11 Z2.5 (m)": "z2pt5",

    'U_channel_code': None,
    'W_channel_code': None,
    'V_channel_code': None,
    'U_hp': None,
    'V_hp': None,
    'W_hp': None,
    'U_lp': None,
    'V_lp': None,
    'W_lp': None,

    'rotD50_pga': 'PGA',
}


def accept_file(file_path) -> bool:
    """Tell whether the given source file can be accepted as waveform file

    :param file_path: the scanned file absolute path (it can also be a file within a zip
        file, in that case the parent directory name is the zip file name)
    """
    return splitext(file_path)[1].startswith('.ASC')


def pre_process(metadata: pd.DataFrame) -> pd.DataFrame:
    """Pre-process the metadata Dataframe. This is usually the place where the given
    dataframe is setup in order to easily find records from file names, or optimize
    some column data (e.g. convert strings to categorical).

    :param metadata: the metadata DataFrame. The DataFrame columns come from the global
        `source_metadata_fields` dict, using each value if not None, otherwise its key.

    :return: a pandas DataFrame optionally modified from `metadata`
    """
    cols = ["network_code", "station_code", "location_code", "instrument_code"]
    for c in cols:
        metadata[c] = metadata[c].astype(str)
    metadata = metadata.dropna(subset=cols + ['event_id'])
    metadata['station_id'] = metadata[cols].agg('.'.join, axis=1)
    metadata = metadata.drop(columns=cols)

    metadata['event_id'] = metadata['event_id'].astype(str).astype('category')
    metadata['station_id'] = metadata['station_id'].astype('category')

    metadata['magnitude_type'] = 'Mw'
    mag_missing = metadata['magnitude'].isna()
    metadata.loc[mag_missing, 'magnitude_type'] = None
    cols = ['Mw', 'Ms', 'ML']
    for mag_type in cols:
        mag_to_be_set = mag_missing & metadata[mag_type].notna()
        if mag_to_be_set.any():
            metadata.loc[mag_to_be_set, 'magnitude'] = metadata[mag_type][mag_to_be_set]
            metadata.loc[mag_to_be_set, 'magnitude_type'] = mag_type
            mag_missing = mag_missing & (~mag_to_be_set)
    metadata = metadata.drop(columns=cols)

    metadata['origin_time'] = pd.to_datetime(metadata['origin_time'])
    metadata['origin_time_resolution'] = 's'

    fault_types = {
        'SS': 'Strike-Slip',
        'NF': 'Normal',
        'TF': 'Reverse',
        'O': 'Normal-Oblique'
    }
    metadata.loc[~metadata['fault_type'].isin(fault_types.keys()), 'fault_type'] = None
    for key, repl in fault_types.items():
        metadata.loc[metadata['fault_type'] == key, 'fault_type'] = repl

    metadata['vs30measured'] = ~pd.isna(metadata.pop('vs30_meas_type'))
    metadata.loc[pd.notna(metadata['vs30']), 'vs30measured'] = True
    vs30_wa = metadata.pop('vs30_m_sec_WA')
    set_vs30 = pd.isna(metadata['vs30']) & pd.notna(vs30_wa)
    metadata.loc[set_vs30, 'vs30'] = vs30_wa[set_vs30]
    metadata.loc[set_vs30, 'vs30measured'] = False

    metadata["lower_cutoff_frequency_h1"] = np.nan
    metadata["lowest_usable_frequency_h1"] = np.nan
    metadata["lower_cutoff_frequency_h2"] = np.nan
    metadata["lowest_usable_frequency_h2"] = np.nan
    metadata["upper_cutoff_frequency_h1"] = np.nan
    metadata["upper_cutoff_frequency_h2"] = np.nan

    cols = {
        'U_channel_code': ('U_hp', 'U_lp'),
        'V_channel_code': ('V_hp', 'V_lp'),
        'W_channel_code': ('W_hp', 'W_lp')
    }
    for ch_code_col, (hp_col, lp_col) in cols.items():
        hp_values = metadata[hp_col]
        lp_values = metadata[lp_col]

        north_south = metadata[ch_code_col] == 'N'
        metadata.loc[north_south, "lower_cutoff_frequency_h1"] = hp_values[north_south]
        metadata.loc[north_south, "higher_cutoff_frequency_h1"] = lp_values[north_south]
        metadata.loc[north_south, "lowest_usable_frequency_h1"] = hp_values[north_south]

        east_west = metadata[ch_code_col] == 'E'
        metadata.loc[east_west, "lower_cutoff_frequency_h2"] = hp_values[east_west]
        metadata.loc[east_west, "higher_cutoff_frequency_h2"] = lp_values[east_west]
        metadata.loc[east_west, "lowest_usable_frequency_h2"] = hp_values[east_west]

        metadata = metadata.drop(columns=[ch_code_col, hp_col, lp_col])

    metadata['PGA'] = metadata['PGA'] / 100  # from cm/sec2 to m/sec2

    metadata = metadata.set_index(['event_id', 'station_id'], drop=True)
    return metadata


def find_sources(file_path: str, metadata: pd.DataFrame) \
        -> tuple[Optional[str], Optional[str], Optional[str], Optional[pd.Series]]:
    """Find the file paths of the three waveform components, and their metadata

    :param file_path: the waveform path currently processed. it is one of the files
        accepted via `accept_file` and it should denote one of the three waveform
        components (the other two should be inferred from it)
    :param metadata: the Metadata dataframe. The returned waveforms metadata must be one
        row of this object as pandas Series (any other object will raise)

    :return: A tuple with three strings denoting the file absolute paths of the three
        components (horizontal1, horizontal2, vertical, in **this order** and the
        pandas Series denoting the waveforms metadata (common to the three components)
    """
    ev_id = splitext(basename(dirname(file_path)))[0]
    sta_id = ".".join(basename(file_path).split('.')[:4])
    file_suffix = basename(file_path).removeprefix(sta_id)
    orientation = sta_id[-1]
    sta_id = sta_id[:-1]
    if orientation in {'N', 'E', 'Z'}:
        try:
            meta = metadata.loc[(ev_id, sta_id)]
        except KeyError:
            meta = pd.Series()

        file_path_n = join(dirname(file_path), f'{sta_id}N{file_suffix}')
        file_path_e = join(dirname(file_path), f'{sta_id}E{file_suffix}')
        file_path_z = join(dirname(file_path), f'{sta_id}Z{file_suffix}')

        if orientation == 'N':
            return file_path, file_path_e, file_path_z, meta
        elif orientation == 'E':
            return file_path_n, file_path, file_path_z, meta
        else:
            return file_path_n, file_path_e, file_path, meta

    return None, None, None, None


def read_waveform(file_path: str, content: BytesIO, metadata: pd.Series) -> Waveform:
    """Read a waveform from a file path

    :param file_path: the waveform path currently processed. It is one of the files
        accepted via `accept_file` and it should denote one of the three waveform
        components. You do not need to open the file here (see `content` parameter)
    :param content: a BytesIO (file-like) object with the content of file_path, as byte
        sequence
    :param metadata: the pandas Series related to the given file, as returned from
        `find_sources`

    :return: a `Waveform` object
    """
    factor = None
    dt = None
    pos = 0
    for line in content:
        pos = content.tell()  # remember current position
        line = line.strip()
        if b':' not in line:
            break
        line = line.decode('utf8')
        key, val = line.split(':', 1)
        key, val = key.strip(), val.strip()
        if key == 'DATA_TYPE':
            assert val == 'ACCELERATION', f'Invalid data type: {val}'
        elif key == 'UNITS':
            assert val in ('cm/s^2', 'm/s^2', 'g'), f'Invalid unit: {val}'
            if val == 'cm/s^2':
                factor = 0.01
            elif val == 'g':
                factor = 9.80665
        elif key == 'SAMPLING_INTERVAL_S':
            dt = float(val)

        if 'DATA_CITATION' in key or 'DATA_CREATOR' in key or 'DATA_MEDIATOR' in key:
            continue
        metadata[f'.{key}'] = val

    assert dt is not None, 'dt not found in file'

    content.seek(pos)
    # Load data into numpy array from that line onward
    data = np.fromstring(content.read().decode('utf-8'), sep='\n', dtype=float)
    if factor is not None:
        data *= factor

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
    """
    Custom post-processing on the metadata and waveforms read from disk.
    Typically, you complete metadata and waveforms, e.g. filling the former with missing
    fields, or converting the latter to the desired units (m/sec*sec, m/sec, m).
    **Remember** that Waveform objects are IMMUTABLE, so you need to return new
    Waveform object if modified

    :param metadata: the pandas Series related to the given file, as returned from
        `find_sources`. Non-standard fields do not need to be removed, missing standard
        fields will be filled with defaults (NaN, None or anything implemented in
        `metadata_fields.yml`)
    :param h1: the Waveform of the first horizontal component, or None (waveform N/A)
    :param h2: the Waveform of the second horizontal component, or None (waveform N/A)
    :param v: the Waveform of the vertical component, or None (waveform N/A)
    """
    # metadata contains also the entries below (PREFIXED WITH A DOT to avoid conflicts)
    # stored in the data file, EXCEPT the entries whose keys contain either
    # DATA_CITATION, DATA_CREATOR, DATA_MEDIATOR (verbose and unnecessary):

    # EVENT_NAME: None
    # EVENT_ID: TK-2000-0449
    # EVENT_DATE_YYYYMMDD: 20000823
    # EVENT_TIME_HHMMSS: 134126
    # EVENT_LATITUDE_DEGREE: 40.7820
    # EVENT_LONGITUDE_DEGREE: 30.7600
    # EVENT_DEPTH_KM: 10.5
    # HYPOCENTER_REFERENCE: ISC-webservice
    # MAGNITUDE_W: 5.2
    # MAGNITUDE_W_REFERENCE: Pondrelli_et_al_2002_dataset
    # MAGNITUDE_L:
    # MAGNITUDE_L_REFERENCE:
    # FOCAL_MECHANISM: Strike-slip faulting
    # NETWORK: TK
    # STATION_CODE: 1001
    # STATION_NAME: AI_146_BLK
    # STATION_LATITUDE_DEGREE: 39.650030
    # STATION_LONGITUDE_DEGREE: 27.856860
    # STATION_ELEVATION_M:
    # LOCATION: 00
    # SENSOR_DEPTH_M:
    # VS30_M/S:
    # SITE_CLASSIFICATION_EC8: B (inferred from topography)
    # MORPHOLOGIC_CLASSIFICATION:
    # EPICENTRAL_DISTANCE_KM: 277.2
    # EARTHQUAKE_BACKAZIMUTH_DEGREE: 244.0
    # DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS: 20000823_134250.000
    # DATE_TIME_FIRST_SAMPLE_PRECISION: seconds
    # SAMPLING_INTERVAL_S: 0.005000
    # NDATA: 18904
    # DURATION_S: 94.520
    # STREAM: HNE
    # UNITS: cm/s^2
    # INSTRUMENT: HN
    # INSTRUMENT_ANALOG/DIGITAL: D
    # INSTRUMENTAL_FREQUENCY_HZ:
    # INSTRUMENTAL_DAMPING:
    # FULL_SCALE_G:
    # N_BIT_DIGITAL_CONVERTER:
    # PGA_CM/S^2: 1.326285
    # TIME_PGA_S: 30.350000
    # BASELINE_CORRECTION: BASELINE REMOVED
    # FILTER_TYPE: BUTTERWORTH
    # FILTER_ORDER: 2
    # LOW_CUT_FREQUENCY_HZ: 0.200
    # HIGH_CUT_FREQUENCY_HZ: 20.000
    # LATE/NORMAL_TRIGGERED: NT
    # DATABASE_VERSION: 0.5
    # HEADER_FORMAT: DYNA 1.2
    # DATA_TYPE: ACCELERATION
    # PROCESSING: manual (Paolucci et al., 2011)
    # DATA_TIMESTAMP_YYYYMMDD_HHMMSS: 20250806_133936.520
    # DATA_LICENSE: U (unknown license)
    # DATA_CITATION: <NOT PRESENT (see above)>
    # DATA_CREATOR: <NOT PRESENT (see above)>
    # ORIGINAL_DATA_MEDIATOR_CITATION: <NOT PRESENT (see above)>
    # ORIGINAL_DATA_MEDIATOR: <NOT PRESENT (see above)>
    # ORIGINAL_DATA_CREATOR_CITATION: <NOT PRESENT (see above)>
    # ORIGINAL_DATA_CREATOR: network: <NOT PRESENT (see above)>

    not_na = pd.notna
    is_na = pd.isna

    # process remaining data:
    if is_na(metadata.get('station_id')):
        metadata['station_id'] = ".".join([
            metadata['.NETWORK'],
            metadata['.STATION_CODE'],
            metadata['.LOCATION'],
            metadata['.INSTRUMENT']
        ])

    if 'filter_type' not in metadata:
        if metadata['.FILTER_TYPE'] == 'BUTTERWORTH':
            metadata['filter_type'] = 'A'
            metadata['filter_order'] = int(metadata['.FILTER_ORDER'] or 0)

    low_cutoff = metadata.get('.LOW_CUT_FREQUENCY_HZ')
    high_cutoff = metadata.get('.HIGH_CUT_FREQUENCY_HZ')
    if metadata['.STREAM'][2] == 'N':
        if is_na(metadata['lower_cutoff_frequency_h1']) and not_na(low_cutoff):
            metadata['lower_cutoff_frequency_h1'] = low_cutoff
            metadata['lowest_usable_frequency_h1'] = low_cutoff
        if is_na(metadata['upper_cutoff_frequency_h1']) and not_na(high_cutoff):
            metadata['upper_cutoff_frequency_h1'] = high_cutoff
    if metadata['.STREAM'][2] == 'E':
        if is_na(metadata['lower_cutoff_frequency_h2']) and not_na(low_cutoff):
            metadata['lower_cutoff_frequency_h2'] = low_cutoff
            metadata['lowest_usable_frequency_h2'] = low_cutoff
        if is_na(metadata['upper_cutoff_frequency_h2']) and not_na(high_cutoff):
            metadata['upper_cutoff_frequency_h2'] = high_cutoff

    if is_na(metadata.get('magnitude')):
        if not_na(metadata.get('.MAGNITUDE_W')):
            metadata['magnitude'] = metadata['.MAGNITUDE_W']
            metadata['magnitude_type'] = 'Mw'
        elif not_na(metadata.get('.MAGNITUDE_L')):
            metadata['magnitude'] = metadata['.MAGNITUDE_L']
            metadata['magnitude_type'] = 'ML'

    if 'fault_type' not in metadata and metadata.get('.FOCAL_MECHANISM'):
        metadata['fault_type'] = metadata['.FOCAL_MECHANISM'].removesuffix(' faulting')

    if is_na(metadata.get('start_time')) and \
            not_na(metadata.get('.DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS')):
        for fmt in ("%Y%m%d_%H%M%S.%f", "%Y%m%d_%H%M%S"):
            try:
                metadata['start_time'] = datetime.strptime(
                    metadata['.DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS'],
                    fmt
                )
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Cannot parse start_time from waveform file: "
                             f"{metadata['.DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS']}")

    if is_na(metadata.get('PGA')) and metadata.get('.PGA_CM/S^2'):
        metadata['PGA'] = metadata['.PGA_CM/S^2'] / 100

    if is_na(metadata.get('origin_time')) and metadata.get('.EVENT_DATE_YYYYMMDD'):
        date = metadata['.EVENT_DATE_YYYYMMDD']
        metadata['origin_time'] = datetime(
            year=int(date[:4]),
            month=int(date[4:6]),
            day=int(date[6:]),
            hour=0, minute=0, second=0, microsecond=0
        )
        metadata['origin_time_resolution'] = 'D'
        if metadata.get('.EVENT_TIME_HHMMSS'):
            dtime = metadata['.EVENT_TIME_HHMMSS']
            metadata['origin_time_resolution'] = 's'
            metadata['origin_time'] = datetime(
                year=date.year,
                month=date.month,
                day=date.day,
                hour=int(dtime[:2]),
                minute=int(dtime[2:4]),
                second=int(dtime[4:6]),
                microsecond=0
            )

    for key, new_key in {
        'EVENT_ID': 'event_id',
        'EVENT_LATITUDE_DEGREE': 'event_latitude',
        'EVENT_LONGITUDE_DEGREE': 'event_longitude',
        'EVENT_DEPTH_KM': 'event_depth',
        'STATION_LATITUDE_DEGREE': 'station_latitude',
        'STATION_LONGITUDE_DEGREE': 'station_longitude',
        'STATION_ELEVATION_M': 'station_height',
        # 'SENSOR_DEPTH_M': None,  # FIXME CHECK
        'VS30_M/S': 'vs30',
        # 'SITE_CLASSIFICATION_EC8': None,
        'EPICENTRAL_DISTANCE_KM': 'epicentral_distance',
        'DATE_TIME_FIRST_SAMPLE_YYYYMMDD_HHMMSS': 'start_time',
    }.items():
        if is_na(metadata.get(new_key)):
            metadata[new_key] = metadata[f".{key}"]

    if is_na(metadata.get('vs30')) and not_na(metadata.get('SITE_CLASSIFICATION_EC8')):
        val = {
            "A": 900,
            "B": 580,
            "C": 270,
            "D": 150,
            "E": 100
        }.get(metadata['SITE_CLASSIFICATION_EC8'])
        if not_na(val):
            metadata['vs30'] = val
            metadata['vs30measured'] = False

    if not_na(metadata.get('epicentral_distance')) and \
            not_na(metadata.get('event_depth')) and \
            is_na(metadata.get('hypocentral_distance')):
        metadata['hypocentral_distance'] = np.sqrt(
            (metadata['epicentral_distance'] ** 2) + metadata['event_depth'] ** 2
        )

    return metadata, h1, h2, v


###########################################
# The code below should not be customized #
###########################################


def main():  # noqa
    """main processing routine called from the command line"""
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
    raise_if_file_exists = False
    if existing:
        res = input(
            f'\nSome destination data already exists. Type:'
            f'\ny: delete and re-create all data'
            f'\nm: delete and re-create metadata, save only new waveform files'
            f'\nAny key: quit\n'
        )
        if res not in ('y', 'm'):
            sys.exit(1)
        if res == 'y':
            raise_if_file_exists = True  # no accidental writing files with same name
            if isdir(dest_waveforms_path):
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

    mandatory_fields = [
        m for m in metadata_fields
        if '[mandatory]' in metadata_fields[m].get('help', '').lower()
    ]

    print(f'Scanning source waveforms directory...', end=" ", flush=True)
    files = scan_dir(source_waveforms_path)
    print(f'{len(files):,} file(s) found')

    print(f'Reading source metadata file...', end=" ", flush=True)
    csv_args: dict[str, Any] = dict(source_metadata_csv_args)
    # csv_args.setdefault('chunksize', 10000)
    csv_args.setdefault(
        'usecols', csv_args.get('usecols', {}) | source_metadata_fields.keys()
    )
    metadata = pd.read_csv(source_metadata_path, **csv_args)
    metadata = metadata.rename(
        columns={k: v for k, v in source_metadata_fields.items() if v is not None}
    )
    old_len = len(metadata)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.SettingWithCopyWarning)
        metadata = pre_process(metadata).copy()
    if len(metadata) < old_len:
        logging.warning(f'{old_len - len(metadata)} metadata row(s) '
                        f'removed in pre-processing stage')
    print(f'{len(metadata):,} record(s), {len(metadata.columns):,} field(s) per record, '
          f'{old_len - len(metadata)} row(s) removed')

    print(f'Creating harmonized dataset from source')
    pbar = tqdm(
        total=len(files),
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "(estimated remaining time {remaining}s)"
    )
    records = []
    item_num = 0
    errs = 0
    saved_waveforms = 0
    total_waveforms = 0
    while len(files):
        num_files = 1
        file = files.pop()

        try:
            h1_path, h2_path, v_path, record = find_sources(file, metadata)

            # checks:
            if sum(_ is not None for _ in (h1_path, h2_path, v_path)) == 0:
                raise Exception('No existing file found')
            if not isinstance(record, pd.Series):
                if isinstance(record, pd.DataFrame):
                    raise Exception('Multiple metadata record found')
                raise Exception('No metadata record found')
            record = record.copy()
            for _ in (h1_path, h2_path, v_path):
                if _ in files:
                    num_files += 1
                    files.remove(_)

            comps = {}
            for cmp_name, cmp_path in zip(('h1', 'h2', 'v'), (h1_path, h2_path, v_path)):
                comps[cmp_name] = None
                if cmp_path:
                    try:
                        with open_file(cmp_path) as file_p:
                            comps[cmp_name] = read_waveform(cmp_path, file_p, record)
                    except OSError:
                        pass
            if all(_ is None for _ in comps.values()):
                raise Exception('No waveform read')
            if len(set(_.dt for _ in comps.values() if _ is not None)) != 1:
                raise Exception('Waveform components have mismatching dt')

            # process waveforms
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
                dtype = metadata_fields[f]['dtype']
                try:
                    clean_record[f] = cast_dtype(val, dtype)
                    if f in mandatory_fields and pd.isna(clean_record[f]):
                        val = 'N/A'
                        raise AssertionError()
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

            # save waveforms
            file_path = join(dest_waveforms_path, get_file_path(clean_record))
            total_waveforms += 1
            if not isfile(file_path):
                save_waveforms(file_path, h1, h2, v)
                saved_waveforms += 1
            elif raise_if_file_exists:
                raise ValueError(f'Waveforms file already exists: {file_path}')

            # save metadata:
            records.append(clean_record)
            if len(records) > 1000:
                save_metadata(
                    dest_metadata_path,
                    pd.DataFrame(records),
                    metadata_fields
                )
                records = []

        except Exception as exc:
            fname, lineno = exc_func_and_lineno(exc, __file__)
            logging.error(f"{exc}. File: {file}. Function {fname}, line {lineno}")
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
    msg = f'Dataset created: {total_waveforms} waveform(s), ' \
          f'({saved_waveforms} newly created)'
    print(msg)
    logging.info(msg)
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
        assert isfile(data['source_metadata']), \
            f"'source_metadata' is not a file: {data['source_metadata']}"
        assert isdir(data['source_data']), \
            f"'source_data' is not a directory: {data['source_data']}"
        return data['source_metadata'], data['source_data'], data['destination']
    except Exception as exc:
        raise ValueError(f'Yaml error ({basename(yaml_path)}): {exc}')


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
            try:
                with zipfile.ZipFile(file_path, 'r') as z:
                    for name in z.namelist():
                        file_path2 = join(file_path, name)
                        if accept_file(file_path2):
                            files.add(file_path2)
            except zipfile.BadZipFile as exc:
                logging.info(f'Skipping bad zip file: {file_path}')
                pass
    return files


def open_file(file_path) -> BytesIO:
    """
    Open a regular file or a file inside a .zip archive. file_path is any item
    returned by `scan_dir`
    """
    fp_lower = file_path.lower()
    try:
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
                    raise OSError(
                        f"{file_path} contains {len(namelist)} files, expected one only")
                data = z.read(namelist[0])
        else:
            with open(file_path, "rb") as f:
                data = f.read()
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile, KeyError) as e:
        raise OSError(f"Failed to read {file_path}: {e}") from e

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
            field_dtype = field['dtype']
            if isinstance(field_dtype, (list, tuple)):
                assert 'default' not in field or field['default'] in field_dtype
                field['dtype'] = pd.CategoricalDtype(field_dtype)
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

    for t_before, t_after in [
        ('origin_time', 'p_wave_arrival_time'),
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


def exc_func_and_lineno(exc, module_path: str = __file__) -> tuple[str, int]:
    """
    Return the innermost function name and line number within `__file__`
    that raised `exc`
    """
    tb = exc.__traceback__
    deepest = None

    while tb:
        frame = tb.tb_frame
        filename = frame.f_code.co_filename

        if os.path.samefile(filename, module_path):
            deepest = frame

        tb = tb.tb_next

    # fallback to outermost frame if none found
    if deepest is None:
        deepest = exc.__traceback__.tb_frame

    return deepest.f_code.co_name, deepest.f_lineno


if __name__ == "__main__":
    main()
