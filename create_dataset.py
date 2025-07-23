"""
Template for the generation of a Time histories database. Steps for the generation:
1. Copy this file as well as metadata_fields in a empty directory.
   You can leave empty cells if data is N/A or missing.
2. Edit this file:
   2a. Set your souece metadata csv file (variable source_metadata_path)
   2b. How to read time histories from it (functions get_waveforms_path and read_waveform)
   2c. How to process time histories and potentially modify the associated CSV row
       (function process_waveforms)
3. Eventually, execute this file *on the terminal within the Python virtual environment*
   (or Conda env)
   `python3 create_dataset.py`
   (will scan all rows of your source metadata, process them and put them in the
   `waveforms` sub-directory of this root directory, creating a new metadata file)
"""
import os
from os.path import abspath, join, basename, isdir, isfile
from os import makedirs
import csv
import json
import sys
import fnmatch
import obspy
from obspy import Trace, read
from datetime import datetime
import pandas as pd
import numpy as np
import scipy
import glob
from tqdm import tqdm


source_metadata_path:str = "path/to/my/source/metadata.csv"  # Your metadata CSV file:


def get_waveforms_path(metadata_row: dict) -> tuple[str, str, str]:
    """Get the full source paths of the waveforms (h1, h2 and v components, in this
    order) from the given metadata row `m_row`.
    Paths can be empty or None, meaning that the relative file is missing. This has to
    be taken into account in `process_waveforms` in case (see below)

    :param metadata_row: dict corresponding to a row of your source metadata. Each dit
        key represents a Field (or Column). Note that float, str, datetime and
        categorical values can also be None (e.g., if the Metadata cell was empty)
    """
    # get the path stored in the metadata file (just an example):
    h1_path = metadata_row['fpath_h1']
    h2_path = metadata_row['fpath_h2']
    v_path = metadata_row['fpath_v']

    # find file? build path? Example:
    full_paths = [
        "?", # compose your source path of h1 maybe using h1_path?
        "?", # compose your source path of h2, maybe using h2_path?
        "?"  # compose your source path of v, maybe using v_path?
    ]
    return (read_waveform(_) for _ in full_paths)


def read_waveform(full_abs_path: str) -> Trace:
    """Read a waveform from a file path. Modify according to the format you stored
    your time histories"""
    return obspy.read(full_path, format='KNET')[0]


def process_waveforms(
        metadata_row: dict, h1: Trace, h2: Trace, v: Trace
) -> tuple[dict, Optional[Trace], Optional[Trace], Optional[Trace]]:
    """Process the waveform(s), returning the same argument modified according to your
    custom processing routine

    :param metadata_row: dict corresponding to a row of your source metadata. Each dit
        key represents a Field (or Column). Note that float, str, datetime and
        categorical values can also be None (e.g., if the Metadata cell was empty).
        You can modify this dict if you want, in case the modifications will be saved in
        the destination metadata file. Note however that any new key will not be saved
    :param h1: first horizontal component, as obspy Trace, or None (no Trace)
    :param h2: second horizontal component, as obspy Trace, or None (no Trace)
    :param v: vertical component, as obspy Trace, or None (no Trace)
    """
    # simply return the arguments (no processing by default):
    return metadata_row, h1, h2, v


#########################################
# The code below should not be customized
#########################################


def save_waveforms(root_path, record, h1, h2, v):
    if h1 is not None:
        save_waveform(h1, abspath(join(root_path, record['fpath_h1'])))
    if h2 is not None:
        save_waveform(h2, abspath(join(root_path, record['fpath_h2'])))
    if v is not None:
        save_waveform(v, abspath(join(root_path, record['fpath_v'])))


def save_waveform(trace, dest_path):
    os.makedirs(os.path.dirname(dest_path))
    Stream([trace]).write(dest_path, format='MSEED')
    # Add read permission for group (stat.S_IRGRP) and others (stat.S_IROTH).
    os.chmod(
        dest_path,
        os.stat(dest_path).st_mode | stat.S_IRGRP | stat.S_IROTH
    )


def _cast_dtype(val: Any, dtype:str):
    if dtype == 'float':
        assert val is None or isinstance(val, float)
    elif dtype == 'int':
        assert isinstanve(val, int)
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
            f'({basename(dest_wavefrom_path)}) already exist in {dest_root_path}.\n'
            f'If you type "y", Metadata file will be deleted and recreated, and '
            f'waveforms files potentially overwritten.\n'
            f'Proceed (y=yes, any key=no)?'
        )
        if res != 'y':
            sys.exit(1)

    if isifile(dest_metadata_path):
        os.unlink(dest_metadata_path)

    # sanitize the metadata
    metadata_fields:dict = yaml.safe_load("./metadata_fields.yml")

    smp = source_metadata_path
    with open(smp, 'rb') as f:
        max_rows = sum(1 for _ in f) - 1  # Subtract 1 for header
    pbar = tqdm(
        total=max_rows,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "(estimated remaining time {remaining}s)"
    )

    rec_num = 0
    for i, metadata in pd.read_csv(smp, chunksize=100000):
        for record in metadata.itertuples(index=False):
            rec_num += 1
            pbar.update(1)

            record = record._asdict()
            for f in metdata_fields.items():
                dtype = metadata_fields[f]['dtype']
                try:
                    record[f] = _cast_dtype(record[f], dtype)
                except AssertionError:
                    print(f"Error in (metadata row #{rec_num}): '{f}' "
                          f"should be {dtype}",
                          file=sys.stderr)

            try:
                h1, h2, v = get_waveform_paths(record)
            except Exception as exc:
                print(f"Error in get_waveform_paths (metadata row #{rec_num}): {exc}",
                      file=sys.stderr)
                sys.exit(1)

            try:
                record, h1, h2, v = process_waveforms(
                    record,
                    read_waveform(h1) if h1 else None,
                    read_waveform(h1) if h2 else None,
                    read_waveform(h1) if v else None
                )
            except Exception as exc:
                print(f"Error in process_waveforms (metadata row #{rec_num}): {exc}",
                      file=sys.stderr)
                sys.exit(1)

            # check returned metadata
            for f in metdata_fields.items():
                dtype = metadata_fields[f]['dtype']
                try:
                    record[f] = _cast_dtype(record[f], dtype)
                except AssertionError:
                    print(f"Error in metadata row #{rec_num} after processing: "
                          f"'{f}' should be {dtype}",
                          file=sys.stderr)

            # save waveforms
            try:
                save_waveforms(dest_waveforms_path, record, h1, h2, v)
            except Exception as exc:
                print(f"Error in save_waveforms (metadata row #{rec_num}): {exc}",
                      file=sys.stderr)
            sys.exit(1)

        # save metadata:
        metadata.to_csv(
            dest_metadata_path,
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
