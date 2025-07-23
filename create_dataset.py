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
from os.path import abspath, join, basename, isdir, isfile, dirname
import stat
import yaml
import csv
import json
import sys
import fnmatch
import obspy
from obspy import Trace, Stream
from datetime import datetime
import pandas as pd
import numpy as np
import scipy
import glob
from tqdm import tqdm


########################################################################
# Editable file part: please read carefully and implement your routine #
########################################################################

source_metadata_path:str = "path/to/my/source/metadata.csv"  # Your metadata CSV file:


def get_waveforms_path(metadata_row: dict) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Get the full source paths of the waveforms (h1, h2 and v components, in this
    order) from the given metadata row `m_row`.
    Paths can be empty or None, meaning that the relative file is missing. This has to
    be taken into account in `process_waveforms` in case (see below). If files are not
    missing, then the file must exist

    :param metadata_row: dict corresponding to a row of your source metadata. Each dict
        key represents a Metadata Field (Column). Note that float, str, datetime and
        categorical values can also be None (e.g., if the Metadata cell was empty)
    """
    # get the path stored in the metadata file (just an example):
    h1_path = metadata_row['fpath_h1']
    h2_path = metadata_row['fpath_h2']
    v_path = metadata_row['fpath_v']

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
    # below, files must exist and the routine will break otherwise
    return (
        None if not isfile(files[0]) else files[0],
        None if not isfile(files[1]) else files[1],
        None if not isfile(files[2]) else files[2]
    )


def read_waveform(full_abs_path: str) -> Trace:
    """Read a waveform from a file path. Modify according to the format you stored
    your time histories"""
    return obspy.read(full_abs_path, format='KNET')[0]


def process_waveforms(
        metadata_row: dict, h1: Trace, h2: Trace, v: Trace
) -> tuple[dict, Optional[Trace], Optional[Trace], Optional[Trace]]:
    """Process the waveform(s), returning the same argument modified according to your
    custom processing routine. Please remember to provide waveforms in standard units
    (m/sec*sec, m/sec, m) and consistent with the value of metadata_row['sensor_type']
    ('A', 'V', 'D').

    :param metadata_row: dict corresponding to a row of your source metadata. Each dict
        key represents a Metadata field (Column). Note that float, str, datetime and
        categorical values can also be None (e.g., if the Metadata cell was empty).
        You can modify this dict if you want, in case the modifications will be saved in
        the destination metadata file. Note however that any new key will not be saved
    :param h1: first horizontal component, as obspy Trace, or None (no Trace)
    :param h2: second horizontal component, as obspy Trace, or None (no Trace)
    :param v: vertical component, as obspy Trace, or None (no Trace)
    """
    # simply return the arguments (no processing by default):
    return metadata_row, h1, h2, v


###########################################
# The code below should not be customized #
###########################################


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

    # sanitize the metadata using asociated yaml:
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
    for i, metadata in pd.read_csv(smp, chunksize=100000):
        new_metadata = []
        for record in metadata.itertuples(index=False):
            rec_num += 1
            pbar.update(1)

            record = record._asdict()
            for f in metadata_fields.items():
                dtype = metadata_fields[f]['dtype']
                try:
                    record[f] = _cast_dtype(record[f], dtype)
                except AssertionError:
                    print(f"Error in (metadata row #{rec_num}): '{f}' "
                          f"should be {dtype}",
                          file=sys.stderr)

            try:
                h1, h2, v = get_waveforms_path(record)
            except Exception as exc:
                print(f"Error in get_waveforms_path (metadata row #{rec_num}): {exc}",
                      file=sys.stderr)
                sys.exit(1)

            # read waveforms separately:
            try:
                h1 = None if h1 is None else read_waveform(h1)
            except Exception as exc:
                print(
                    f"Error in get_waveforms_path (metadata row #{rec_num}) building"
                    f"h1 path. File not found: {h1}", file=sys.stderr
                )
                sys.exit(1)
            try:
                h2 = None if h2 is None else read_waveform(h2)
            except Exception as exc:
                print(
                    f"Error in get_waveforms_path (metadata row #{rec_num}) building"
                    f"h2 path. File not found: {h2}", file=sys.stderr
                )
                sys.exit(1)
                # read waveforms separately:
            try:
                v = None if v is None else read_waveform(v)
            except Exception as exc:
                print(
                    f"Error in get_waveforms_path (metadata row #{rec_num}) building"
                    f"v path. File not found: {v}", file=sys.stderr
                )
                sys.exit(1)

            try:
                record, h1, h2, v = process_waveforms(record, h1, h2, v)
            except Exception as exc:
                print(f"Error in process_waveforms (metadata row #{rec_num}): {exc}",
                      file=sys.stderr)
                sys.exit(1)

            # check returned metadata
            for f in metadata_fields.items():
                dtype = metadata_fields[f]['dtype']
                try:
                    record[f] = _cast_dtype(record[f], dtype)
                except AssertionError:
                    print(f"Error in metadata row #{rec_num} after processing: "
                          f"'{f}' should be {dtype}",
                          file=sys.stderr)
            new_metadata.append(record)

            # save waveforms
            try:
                save_waveforms(dest_waveforms_path, record, h1, h2, v)
            except Exception as exc:
                print(f"Error in save_waveforms (metadata row #{rec_num}): {exc}",
                      file=sys.stderr)
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
