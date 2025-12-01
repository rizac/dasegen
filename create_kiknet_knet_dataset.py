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


class MetadataWriteError(Exception):
    """
    Exception in writing metadata that should be treated specifically (e.g.
    interrupt the code and skip logging)
    """
    pass


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
    # 'header': None  # for CSVs with no header
    # 'dtype': {}  # NOT RECOMMENDED: this might interfere with the default field dtypes
    # 'usecols': []  # NOT RECOMMENDED: this might interfere with the default field names
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
    'RJB_0': 'joyner_boore_distance',
    'RJB_1': 'joyner_boore_distance2',
    'Rrup_0': 'rupture_distance',
    'Rrup_1': 'rupture_distance2',
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
    """Tell whether the given source file can be accepted as waveform file

    :param file_path: the scanned file absolute path (it can also be a file within a zip
        file, in that case the parent directory name is the zip file name)
    """
    return splitext(file_path)[1] in {
        '.UD1', '.NS1', '.EW1', '.UD2', '.NS2', '.EW2', '.UD', '.NS', '.EW'
    }  # with *1 => borehole


def pre_process(metadata: pd.DataFrame) -> pd.DataFrame:
    """Pre-process the metadata Dataframe. This is usually the place where the given
    dataframe is setup in order to easily find records from file names, or optimize
    some column data (e.g. convert strings to categorical).

    :param metadata: the metadata DataFrame. The DataFrame columns come from the global
        `source_metadata_fields` dict, using each value if not None, otherwise its key.

    :return: a pandas DataFrame optionally modified from `metadata`
    """
    metadata = metadata.dropna(subset=['event_id', 'station_id'])
    metadata['event_id'] = metadata['event_id'].astype('category')
    metadata['station_id'] = metadata['station_id'].astype('category')
    metadata = metadata.set_index(["event_id", "station_id"], drop=True)
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

    record: Optional[pd.Series] = None
    try:
        ev_ids = [int(basename(dirname(file_path))), basename(dirname(file_path))]
    except (ValueError, TypeError):
        ev_ids = [basename(dirname(file_path))]
    for ev_id in ev_ids:
        sta_id = basename(root)[:6]  # station name is first 6 letters
        try:
            record = metadata.loc[(ev_id, sta_id)].copy()
            if not isinstance(record, pd.Series):  # safety check
                raise KeyError()
            sta_suffix = f'_{ext[3:4]}' if ext[3:4] else ''
            if sta_suffix:
                sta_id = f'{sta_id}{sta_suffix}'
            record["station_id"] = sta_id
            record["event_id"] = str(ev_id)
        except KeyError:
            continue
        break
    return paths + (record, )


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
    data = np.fromstring(rest, sep=" ", dtype=np.float32)
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
    orig_meta, metadata = metadata, metadata.copy()
    metadata['origin_time'] = datetime.fromisoformat(metadata["origin_time"])
    metadata['origin_time_resolution'] = 's'
    dt_format = "%Y%m%d%H%M%S"
    metadata["start_time"] = \
        datetime.strptime(str(metadata["start_time"]), dt_format)
    metadata["p_wave_arrival_time"] = \
        datetime.strptime(str(metadata["p_wave_arrival_time"]), dt_format)
    metadata["s_wave_arrival_time"] = \
        datetime.strptime(str(metadata["s_wave_arrival_time"]), dt_format)
    metadata['fault_type'] = {
        'S': 'Strike-Slip',
        'N': 'Normal',
        'R': 'Reverse'
    }.get(metadata['fault_type'])
    metadata["vs30measured"] = metadata["vs30measured"] in {1, "1", 1.0}
    metadata["region"] = 0
    metadata["filter_type"] = "A"
    metadata["filter_order"] = 4
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

    dest_log_path = join(dest_root_path, "dataset_creation.log")
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
    with warnings.catch_warnings(record=True) as _w_:
        warnings.simplefilter("always", pd.errors.DtypeWarning)
        metadata = pd.read_csv(source_metadata_path, **csv_args)
        if _w_:
            print(f'({_w_[0].message})', end=" ", flush=True)
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
        bar_format="{percentage:3.0f}%|{bar} {postfix} | ~{remaining}s remaining"
    )
    records = []
    item_num = 0
    errs = 0
    saved_waveforms = 0
    total_waveforms = len(files)
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
            saved_waveforms += 1
            if not isfile(file_path):
                save_waveforms(file_path, h1, h2, v)
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
        except MetadataWriteError:
            raise
        except Exception as exc:
            fname, lineno = exc_func_and_lineno(exc, __file__)
            logging.error(f"{exc}. File: {file}. Function {fname}, line {lineno}")
            errs += 1
        finally:
            saved_ratio = int(100 * saved_waveforms / (total_waveforms/3))
            pbar.set_postfix({
                "saved waveforms": f"{saved_waveforms:,} (~{saved_ratio}%)"
            })
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

    if len(records):
        save_metadata(dest_metadata_path, pd.DataFrame(records), metadata_fields)
    if isfile(dest_metadata_path):
        os.chmod(
            dest_metadata_path,
            os.stat(dest_metadata_path).st_mode | stat.S_IRGRP | stat.S_IROTH
        )

    pbar.close()
    msg = f'Dataset created: {saved_waveforms} waveform(s) ' \
          f'(either already or newly created, depending on settings)'
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
    try:
        if metadata is not None and not metadata.empty:
            # save metadata:
            new_metadata_df = pd.DataFrame(metadata)
            for col in metadata_fields:
                new_metadata_df[col] = cast_dtypes(
                    metadata_fields[col]['dtype'],
                    metadata_fields[col].get('default'),
                    col,
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
            new_metadata_df[list(metadata_fields)].to_hdf(
                dest_metadata_path, **hdf_kwargs
            )
    except Exception as exc:
        raise MetadataWriteError(str(exc))


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
    if isinstance(dtype, str) and values is not None:
        real_dt = str(getattr(values, 'dtype', ''))
        if real_dt.startswith(dtype):
            return values

    if dtype == 'int':
        if values is None:
            return pd.Series(
                np.full(len(dataframe), default_value, dtype=int),
                index=dataframe.index
            )
        return values.astype(int)
    elif dtype == 'bool':
        if values is None:
            return pd.Series(
                np.full(len(dataframe), default_value, dtype=bool),
                index=dataframe.index
            )
        return values.astype(bool)
    elif dtype == 'datetime':
        if values is None:
            if pd.isna(default_value):
                default_value = pd.NaT
            return pd.Series(
                np.full(len(dataframe), default_value),
                index=dataframe.index
            )
        return pd.to_datetime(values, errors='coerce')
    elif dtype == 'str':
        if values is None:
            if pd.isna(default_value):
                default_value = None
            return pd.Series(
                np.full(len(dataframe), default_value, dtype=dtype),
                index=dataframe.index
            )
        return values.astype(str)
    elif dtype == 'float':
        if values is None:
            if pd.isna(default_value):
                default_value = np.nan
            return pd.Series(
                np.full(len(dataframe), default_value, dtype=float),
                index=dataframe.index
            )
        return values.astype(float)
    elif isinstance(dtype, pd.CategoricalDtype):
        if values is None:
            if pd.isna(default_value):
                default_value = None
            return pd.Series(
                np.full(len(dataframe), default_value, dtype=float),
                dtype=dtype, index=dataframe.index
            )
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
