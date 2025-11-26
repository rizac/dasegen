"""
Python script for the generation of a Time histories database.
To create a new database, copy/rename this file and modify the editable
part of this module. See instructions below and README.md fir details
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
min_waveforms_ok_ratio = 1/5

# max discrepancy between PGA from catalog and computed PGA. A waveform is saved if:
# | PGA - PGA_computed | <= pga_retol * | PGA |
pga_retol = 1/4

# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {
    # 'header': None  # for CSVs with no header
    # 'dtype': {}  # NOT RECOMMENDED, see `metadata_fields.yml` instead
    # 'usecols': []  # NOT RECOMMENDED, see `source_metadata_fields` below instead
}

# Mapping from source metadata columns to their new names. Map to None to skip renaming
# and just load the column data
source_metadata_fields = {
    'EQID': "event_id",
    "station_id": "station_id",
    "fpath_h1": None,
    "fpath_h2": None,
    "fpath_v": None,
    'Record Sequence Number': None,

    "EpiD (km)": "epicentral_distance",
    "HypD (km)": "hypocentral_distance",
    "Joyner-Boore Dist. (km)": "joyner_boore_distance",
    "ClstD (km)": "rupture_distance",
    "Rx": "fault_normal_distance",
    'YEAR': None,
    'MODY': None,
    'HRMN': None,
    "Hypocenter Latitude (deg)": "event_latitude",
    "Hypocenter Longitude (deg)": "event_longitude",
    "Hypocenter Depth (km)": "event_depth",
    "Earthquake Magnitude": "magnitude",
    "Magnitude Type": "magnitude_type",
    "Depth to Top Of Fault Rupture Model": "depth_to_top_of_fault_rupture",
    "Fault Rupture Width (km)": "fault_rupture_width",
    "Strike (deg)": "strike",
    "Dip (deg)": "dip",
    "Rake Angle (deg)": "rake",

    "Mechanism Based on Rake Angle": "fault_type",
    "Vs30 (m/s) selected for analysis": "vs30",
    # vs30measured is a boolean expression; treated as key
    "Measured/Inferred Class": "vs30measured",
    "Station Latitude": "station_latitude",
    "Station Longitude": "station_longitude",
    "Northern CA/Southern CA - H11 Z1 (m)": "z1",
    "Northern CA/Southern CA - H11 Z2.5 (m)": "z2pt5",

    "Type of Filter": "filter_type",
    "npass": "npass",
    "nroll": "nroll",
    "HP-H1 (Hz)": "lower_cutoff_frequency_h1",
    "HP-H2 (Hz)": "lower_cutoff_frequency_h2",
    "LP-H1 (Hz)": "upper_cutoff_frequency_h1",
    "LP-H2 (Hz)": "upper_cutoff_frequency_h2",
    "Lowest Usable Freq - H1 (Hz)": "lowest_usable_frequency_h1",
    "Lowest Usable Freq - H2 (H2)": "lowest_usable_frequency_h2",

    "PGA (g)": "PGA"
}


def accept_file(file_path) -> bool:
    """Tell whether the given source file can be accepted as waveform file

    :param file_path: the scanned file absolute path (it can also be a file within a zip
        file, in that case the parent directory name is the zip file name)
    """
    return splitext(file_path)[1].startswith('.AT')


def pre_process(metadata: pd.DataFrame) -> pd.DataFrame:
    """Pre-process the metadata Dataframe. This is usually the place where the given
    dataframe is setup in order to easily find records from file names, or optimize
    some column data (e.g. convert strings to categorical).

    :param metadata: the metadata DataFrame. The DataFrame columns come from the global
        `source_metadata_fields` dict, using each value if not None, otherwise its key.

    :return: a pandas DataFrame optionally modified from `metadata`
    """
    cols = ["fpath_h1", "fpath_h2", "fpath_v"]
    metadata = metadata.dropna(subset=cols + ['event_id', 'station_id'])
    # set event and station categorical (save space)
    metadata['event_id'] = metadata['event_id'].astype('category')
    metadata['station_id'] = metadata['station_id'].astype('category')
    # set fpath as index (+ some prefix addition):
    for col in cols:
        assert not metadata[col].str.startswith('RSN').any()
        metadata[col] = metadata[col].str.strip()
    metadata = metadata.set_index(cols, drop=True)
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
    file_basename = basename(file_path)
    rsn = None
    rsn_prefix = ''
    pattern = re.compile(r'^RSN_?(\d+)_')
    match = pattern.match(file_basename)
    if match:
        rsn = match.group(1)
        rsn_prefix = match.group(0)
        file_basename = file_basename.removeprefix(rsn_prefix)

    root_dir = dirname(file_path)

    for attempt in [
        ([file_basename], slice(None), slice(None)),
        (slice(None), [file_basename], slice(None)),
        (slice(None), slice(None), [file_basename])
    ]:
        try:
            meta = metadata.loc[attempt]  # connot return a Series (slices in loc)
        except KeyError:
            continue
        if len(meta) == 1:
            if rsn is not None:
                if str(meta['Record Sequence Number'].iloc[0]).strip() != rsn:
                    raise ValueError("File name match, RSN doesn't")
            file_names = [rsn_prefix + _ for _ in meta.index[0]]
            return (
                join(root_dir, file_names[0]),
                join(root_dir, file_names[1]),
                join(root_dir, file_names[2]),
                meta.iloc[0]  # convert to Series
            )

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

    # First few lines are headers
    header1 = content.readline().strip()
    header2 = content.readline().strip()
    header3 = content.readline().strip()
    header4 = content.readline().split(b",")
    npts = int(re.match(br"NPTS\s*=\s*(\d+)", header4[0].strip()).group(1))
    dt = float(re.match(br"DT\s*=\s*([\.\d]+)\s*SEC", header4[1].strip()).group(1))
    data_str = b" ".join(line for line in content)
    # The acceleration time series is given in units of g. So I convert it in m/s.
    return Waveform(dt, np.fromstring(data_str, sep=" ") * 9.80665)


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
    # convert time(s):
    year = metadata['YEAR']
    month_day = str(metadata['MODY'])
    if month_day in (-999, '-999'):
        raise AssertionError('Invalid month_day')
    month_day = month_day.zfill(4)  # pad with zeroes
    month, day = int(month_day[:2]), int(month_day[2:])

    hour_min = str(metadata['HRMN'])
    if hour_min in (-999, '-999'):
        evt_time = pd.NaT
        evt_date = datetime(
            year=year, month=month, day=day, hour=0, minute=0, second=0, microsecond=0
        )
    else:
        hour_min = hour_min.zfill(4)  # pad with zeroes
        hour, min = int(hour_min[:2]), int(hour_min[2:])
        evt_time = datetime(year=year, month=month, day=day, hour=hour, minute=min)
        evt_date = evt_time.replace(hour=0, minute=0, second=0, microsecond=0)
    # use datetimes also for event_date (for simplicity when casting later):
    metadata["origin_date"] = evt_date
    metadata["origin_time"] = evt_time

    if metadata["filter_type"] in (-999, '-999'):
        metadata["filter_type"] = None

    if metadata['magnitude_type'] == 'U':
        metadata['magnitude_type'] = None

    try:
        metadata['fault_type'] = [
            'Strike-Slip', 'Normal', 'Reverse', 'Reverse-Oblique', 'Normal-Oblique'
        ][int(metadata['fault_type'])]
    except (IndexError, ValueError, TypeError):
        metadata['fault_type'] = None

    # convert from g to m/s2:
    metadata["PGA"] = metadata["PGA"] * 9.80665

    # vs30 measured has weird values 2a_3a_3b ore whatever. Set to inferred in this case
    metadata['vs30measured'] = metadata['vs30measured'] in {0, '0'}

    # simply return the arguments (no processing by default):
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
    csv_args.setdefault(
        'usecols', csv_args.get('usecols', {}) | source_metadata_fields.keys()
    )
    metadata = pd.read_csv(source_metadata_path, **csv_args)
    metadata = metadata.rename(
        columns={k: v for k, v in source_metadata_fields.items() if v is not None}    # RHB1 and SITE_CLASSIFICATION_EC8  # FIXME DO!
    )
    old_len = len(metadata)
    metadata = pre_process(metadata)
    if len(metadata) < old_len:
        logging.warning(f'{old_len - len(metadata)} metadata row(s) '
                        f'removed in pre-processing stage')
    print(f'{len(metadata):,} record(s), {len(metadata.columns):,} field(s) per record, '
          f'{old_len - len(metadata)} row(s) removed')

    print(f'Creating harmonized dataset from source')
    records = []
    item_num = 0
    errs = 0
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
            for _ in (h1_path, h2_path, v_path):
                if _ in files:
                    num_files += 1
                    files.remove(_)

            comps = {}
            for cmp_name, cmp_path in zip(('h1', 'h2', 'v'), (h1_path, h2_path, v_path)):
                comps[cmp_name] = None
                if cmp_path and isfile(cmp_path):
                    with open_file(cmp_path) as file_p:
                        comps[cmp_name] = read_waveform(cmp_path, file_p, record)

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
            records.append(clean_record)
            if len(records) > 1000:
                save_metadata(dest_metadata_path, pd.DataFrame(records), metadata_fields)
                records = []

            # save waveforms
            file_path = join(dest_waveforms_path, get_file_path(clean_record))
            if not isfile(file_path):
                save_waveforms(file_path, h1, h2, v)

        except Exception as exc:
            fname, lineno = exc_func_and_lineno(exc, __file__)
            logging.error(f"[ERROR] {exc}. In '{fname}', line {lineno}")
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
