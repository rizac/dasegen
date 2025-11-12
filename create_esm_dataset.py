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

from typing import Optional, Any, Union
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


########################################################################
# Editable file part: please read carefully and implement your routine #
########################################################################

# id columns (YAML file names mapped to source catalog names):
id_columns = {
    'event_id': 'event_id',
    'station_id': 'station_id'
}

# csv arguments for source metadata (e/g. 'header'= None)
source_metadata_csv_args = {'sep': ';'}  # {'header': None} for CSVs with no header

# The program will stop if the successfully processed waveform ratio falls below this
# value that must be in [0, 1] (this makes spotting errors and checking log faster):
min_waveforms_ok_ratio = 1/5

# max discrepancy between PGA from catalog and computed PGA. A waveform is saved if:
# | PGA - PGA_computed | <= pga_retol * | PGA |
pga_retol = 1/4


def accept_file(file_path) -> bool:
    """Tell whether the given source file can be accepted as time history file"""
    return splitext(file_path)[1] == '.zip'


def find_waveforms_path(
        metadata: dict,
        waveform_file_paths: dict[str, Union[dict[str], str]]
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Get the full source paths of the waveforms (h1, h2 and v components, in this
    order) from the given row of your source metadata.
    Paths can be empty or None, meaning that the relative file is missing. This has to
    be taken into account in `process_waveforms` in case (see below). If files are not
    missing, then the file must exist

    :param metadata: Python dict, corresponding to a row of your source metadata table.
        Each dict key represents a Metadata Field (Column). Note that float, str,
        datetime and categorical values can also be None (e.g., if the Metadata cell
        was empty)
    :param waveform_file_paths: dict of string denoting the scanned time histories
        source directory. It is a dict of strings denoting directory names or file names.
        If directory names, the mapped value is a nested dict with the same structure,
        otherwise it is the file absolute path
    """
    metadata_paths = [
        ('' if pd.isna(metadata["fpath_h1"]) else metadata["fpath_h1"]).strip(),
        ('' if pd.isna(metadata["fpath_h2"]) else metadata["fpath_h2"]).strip(),
        ('' if pd.isna(metadata["fpath_v"]) else metadata["fpath_v"]).strip()
    ]
    file_paths = [[], [], []]

    rsn = str(metadata['Record Sequence Number'])

    for dir_name in waveform_file_paths:
        for file_abs_path in waveform_file_paths[dir_name].values():
            bname = basename(file_abs_path)
            for i in range(len(metadata_paths)):
                metadata_path = metadata_paths[i]
                if not metadata_path or not bname:
                    continue
                metadata_path = f'{rsn}_{metadata_path}'
                if bname.startswith('RSN_'):
                    metadata_path = f'RSN_{metadata_path}'
                elif bname.startswith('RSN'):
                    metadata_path = f'RSN{metadata_path}'
                if metadata_path == bname:
                    file_paths[i].append(file_abs_path)
                    continue

    return (
        file_paths[0][0] if len(file_paths[0]) == 1 else None,
        file_paths[1][0] if len(file_paths[1]) == 1 else None,
        file_paths[2][0] if len(file_paths[2]) == 1 else None
    )


def read_waveform(full_abs_path: str, metadata: dict) -> tuple[float, ndarray]:
    """Read a waveform from a file path. Modify according to the format you stored
    your time histories"""
    with open(full_abs_path) as f:
        # First few lines are headers
        header1 = f.readline().strip()
        val = f.readline().split(":")[-1].strip()
        metadata.setdefault('event_id', val)

        val = f.readline().split(":")[-1].strip()
        metadata.setdefault('event_id', val)

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
    # pga check
    pga = metadata["PGA (g)"] * 9.80665  # convert m/sec square

    # compute arrival time (correct)
    year = metadata['YEAR']
    month_day = str(metadata['MODY'])
    if month_day in (-999, '-999'):
        raise AssertionError('Invalid month_day')
    month_day = month_day.zfill(4)  # pad with zeroes
    month, day = int(month_day[:2]), int(month_day[2:])

    hour_min = str(metadata['HRMN'])
    if hour_min in (-999, '-999'):
        evt_time = pd.NaT
    else:
        hour_min = hour_min.zfill(4)  # pad with zeroes
        hour, min = int(hour_min[:2]), int(hour_min[2:])
        evt_time = datetime(year=year, month=month, day=day, hour=hour, minute=min)
    # use datetimes also for event_date (for simplicity when casting later):
    evt_date = evt_time.replace(hour=0, minute=0, second=0, microsecond=0)
    evt_id = str(metadata.get('EQID'))
    sta_id = str(metadata["station_id"])

    # """
    # Record Sequence Number,EQID,Earthquake Name,YEAR,MODY,HRMN,Station Name,
    # Station Sequence Number,Station ID  No.,Earthquake Magnitude,Magnitude Type,
    # Magnitude Uncertainty: Kagan Model,
    # Magnitude Uncertainty: Statistical,Magnitude Sample Size,
    # Magnitude Uncertainty: Study Class,Mo (dyne.cm),Strike (deg),Dip (deg),
    # Rake Angle (deg),Mechanism Based on Rake Angle,P-plunge (deg),P-trend (deg),T-plunge (deg),
    # T-trend (deg),Hypocenter Latitude (deg),Hypocenter Longitude (deg),
    # Hypocenter Depth (km),
    # Coseismic Surface Rupture: 1=Yes; 0=No;    -999=Unknown,Coseismic Surface Rupture (Including Inferred),Basis for Inference of Surface Rupture,Finite Rupture Model: 1=Yes;  0=No,
    # Depth to Top Of Fault Rupture Model,Fault Rupture Length for Calculation of Ry (km),
    # Fault Rupture Width (km),Fault Rupture Area (km^2),Avg Fault Disp (cm),Rise Time (s),
    # Avg Slip Velocity (cm/s),Static Stress Drop (bars),Preferred Rupture Velocity (km/s),
    # Average Vr/Vs,Percent of Moment Release in the Top 5 Km of Crust,
    # Existence of Shallow Asperity: 0=No; 1=Yes,Depth to Top of Shallowest Asperity (km),
    # Earthquake in Extensional Regime: 1=Yes; 0=No,Fault Name,Slip Rate (mm/Yr),
    # EpiD (km),HypD (km),Joyner-Boore Dist. (km),Campbell R Dist. (km),RmsD (km),
    # ClstD (km),Rx,FW/HW Indicator,Source to Site Azimuth (deg),X,
    # Theta.D (deg),SSGA (Strike Slip),Y,
    # Phi.D (deg),SSGA (Dip Slip),s,d,ctildepr,Unused Column,D,Rfn.Hyp,Rfp.Hyp,Unused Column,
    # Unused Column,Unused Column,T,GMX's C1,GMX's C2,GMX's C3,Campbell's GEOCODE,Bray and Rodriguez-Marek SGS,
    # Depth,Preferred NEHRP Based on Vs30,Vs30 (m/s) selected for analysis,
    # Column Not Used,Measured/Inferred Class,Sigma of Vs30 (in natural log Units),NEHRP Classification from CGS's Site Condition Map,
    # Geological Unit,Geology,Owner,Station Latitude,Station Longitude,STORIES,
    # INSTLOC,Depth to Basement Rock,Site Visited,NGA Type,Age,Grain Size,Depositional History,
    # Northern CA/Southern CA - H11 Z1 (m),Northern CA/Southern CA - H11 Z1.5 (m),Northern CA/Southern CA - H11 Z2.5 (m),Northern CA/Southern CA - S4 Z1 (m),Northern CA/Southern CA - S4 Z1.5 (m),
    # Northern CA/Southern CA - S4 Z2.5 (m),Depth to Franciscan Rock (km),
    # Basin,h (m),hnorm (m),Rsbe (m),Rcebe (m),Rebe (m),Rsbe1 (m),File Name (Horizontal 1),
    # File Name (Horizontal 2),File Name (Vertical),H1 azimth (degrees),H2 azimith (degrees),
    # Type of Recording,Instrument Model,PEA Processing Flag,
    # Type of Filter,npass,nroll,HP-H1 (Hz),HP-H2 (Hz),LP-H1 (Hz),LP-H2 (Hz),Factor,
    # Lowest Usable Freq - H1 (Hz),Lowest Usable Freq - H2 (H2),Lowest Usable Freq - Ave. Component (Hz),PGA (g),PGV (cm/sec),PGD (cm),T0.010S,T0.020S,T0.022S,T0.025S,T0.029S,T0.030S,T0.032S,T0.035S,T0.036S,T0.040S,T0.042S,T0.044S,T0.045S,T0.046S,T0.048S,T0.050S,T0.055S,T0.060S,T0.065S,T0.067S,T0.070S,T0.075S,T0.080S,T0.085S,T0.090S,T0.095S,T0.100S,T0.110S,T0.120S,T0.130S,T0.133S,T0.140S,T0.150S,T0.160S,T0.170S,T0.180S,T0.190S,T0.200S,T0.220S,T0.240S,T0.250S,T0.260S,T0.280S,T0.290S,T0.300S,T0.320S,T0.340S,T0.350S,T0.360S,T0.380S,T0.400S,T0.420S,T0.440S,T0.450S,T0.460S,T0.480S,T0.500S,T0.550S,T0.600S,T0.650S,T0.667S,T0.700S,T0.750S,T0.800S,T0.850S,T0.900S,T0.950S,T1.000S,T1.100S,T1.200S,T1.300S,T1.400S,T1.500S,T1.600S,T1.700S,T1.800S,T1.900S,T2.000S,T2.200S,T2.400S,T2.500S,T2.600S,T2.800S,T3.000S,T3.200S,T3.400S,T3.500S,T3.600S,T3.800S,T4.000S,T4.200S,T4.400S,T4.600S,T4.800S,T5.000S,T5.500S,T6.000S,T6.500S,T7.000S,T7.500S,T8.000S,T8.500S,T9.000S,T9.500S,T10.000S,T11.000S,T12.000S,T13.000S,T14.000S,T15.000S,T20.000S
    # """
    new_metadata = {
        'event_id': evt_id,
        'epicentral_distance': metadata["EpiD (km)"],
        'hypocentral_distance': metadata["HypD (km)"],
        'joyner_boore_distance': metadata["Joyner-Boore Dist. (km)"],
        'rupture_distance': metadata["ClstD (km)"],
        'fault_normal_distance': metadata['Rx'],
        'origin_time': evt_time,
        'origin_date': evt_date,
        'event_latitude': metadata["Hypocenter Latitude (deg)"],
        'event_longitude': metadata["Hypocenter Longitude (deg)"],
        'event_depth': metadata["Hypocenter Depth (km)"],
        'magnitude': metadata["Earthquake Magnitude"],
        'magnitude_type': metadata["Magnitude Type"],
        'depth_to_top_of_fault_rupture': metadata["Depth to Top Of Fault Rupture Model"],
        'fault_rupture_width': metadata["Fault Rupture Width (km)"],
        'strike': metadata["Strike (deg)"],
        'dip': metadata["Dip (deg)"],
        'rake': metadata["Rake Angle (deg)"],
        'strike2': None,
        'dip2': None,
        'rake2': None,
        'fault_type': metadata["Mechanism Based on Rake Angle"],

        'station_id': sta_id,
        "vs30": metadata["Vs30 (m/s) selected for analysis"],
        "vs30measured": metadata["Measured/Inferred Class"] in {0, "0", 0.0},
        "station_latitude": metadata["Station Latitude"],
        "station_longitude": metadata["Station Longitude"],
        "z1": metadata["Northern CA/Southern CA - H11 Z1 (m)"],
        "z2pt5": metadata["Northern CA/Southern CA - H11 Z2.5 (m)"],
        "region": 0,

        # "sensor_type": 'A',
        "filter_type": metadata["Type of Filter"],
        "npass": metadata["npass"],
        "nroll": metadata["nroll"],
        "lower_cutoff_frequency_h1": metadata["HP-H1 (Hz)"],
        "lower_cutoff_frequency_h2": metadata["HP-H2 (Hz)"],
        "upper_cutoff_frequency_h1": metadata["LP-H1 (Hz)"],
        "upper_cutoff_frequency_h2": metadata["LP-H2 (Hz)"],
        "lowest_usable_frequency_h1": metadata["Lowest Usable Freq - H1 (Hz)"],
        "lowest_usable_frequency_h2": metadata["Lowest Usable Freq - H2 (H2)"],
        'PGA': pga,
    }

    # correct missing values:

    if new_metadata["filter_type"] in (-999, '-999'):
        new_metadata["filter_type"] = None

    if new_metadata['magnitude_type'] == 'U':
        new_metadata['magnitude_type'] = None

    try:
        new_metadata['fault_type'] = [
            'Strike-Slip', 'Normal', 'Reverse', 'Reverse-Oblique', 'Normal-Oblique'
        ][int(new_metadata['fault_type'])]
    except (IndexError, ValueError, TypeError):
        new_metadata['fault_type'] = None

    # simply return the arguments (no processing by default):
    return new_metadata, h1, h2, v


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

    if not isdir(dest_root_path):
        os.makedirs(dest_root_path)

    dest_log_path = join(dest_root_path, basename(__file__) + ".log")
    setup_logging(dest_log_path)

    logging.info(f'Working directory: {abspath(os.getcwd())}')
    logging.info(f'Run command      : {" ".join([sys.executable] + sys.argv)}')
    print(f"Source metadata path: {source_metadata_path}")
    print(f"Source waveforms path: {source_waveforms_path}")

    # Download and save locally
    try:
        dest_metadata_fields_path = join(dest_root_path, 'metadata_fields.yml')
        metadata_fields = get_metadata_fields(dest_metadata_fields_path)
        with open(dest_metadata_fields_path, "r") as _:
            logging.info(f'Metadata fields file: {dest_metadata_fields_path}')
    except Exception as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(f'Scanning {source_metadata_path}')
    min_itemsize = {}
    max_rows = 0
    with open(source_metadata_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            max_rows += 1
            for dest_col, src_col in id_columns.items():
                val = row[src_col]
                if val is not None:
                    old_val = min_itemsize.get(dest_col, 0)
                    min_itemsize[dest_col] = max(old_val, len(val))

    print(f'Scanning {source_waveforms_path}')
    files, num_files = scan_dir(source_waveforms_path)

    print(f'Found {num_files} file(s) as time history candidates')
    pbar = tqdm(
        total=max_rows,
        bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                   "(estimated remaining time {remaining}s)"
    )

    # output metadata to_hdf kwargs (fixed):
    hdf_kwargs = {
        'key': "metadata",  # table name
        'mode': "a",  # (1st time creates a new file because we deleted it, see above)
        'format': "table",  # required for appendable table
        'append': True,  # first batch still uses append=True
        'min_itemsize': min_itemsize,  # required for strings (used only the 1st time to_hdf is called)  # noqa
        # 'data_columns': [],  # list of columns you need to query these later
    }

    print(f'Processing records in {source_metadata_path}')
    rec_num = 0
    csv_args = dict(source_metadata_csv_args)
    csv_args.setdefault('chunksize', 10000)
    errs = 0
    for metadata_chunk in pd.read_csv(source_metadata_path, **csv_args):
        new_metadata = []
        for record in metadata_chunk.to_dict(orient="records"):

            rec_num += 1
            metadata_row = f'metadata row #{rec_num}'
            pbar.update(1)

            step_name = ""
            components = {'h1': None, 'h2': None, 'v': None}
            h1_path, h2_path, v_path = None, None, None
            try:
                step_name = "find_waveform_paths"
                h1_path, h2_path, v_path = find_waveforms_path(record, files)

                if (h1_path is None) and (h2_path is None) and (v_path is None):
                    raise Exception('No waveform file')

                for comp_name in components:
                    step_name = f"read_waveform ({comp_name})"
                    comp_path = {'h1': h1_path, 'h2': h2_path, 'v': v_path}[comp_name]
                    if comp_path:
                        components[comp_name] = read_waveform(comp_path, record)

                if all(_ is None for _ in components.values()):
                    raise Exception('No waveform read')
                # elif any(_ is None for _ in components.values()):
                #     logging.warning(
                #         f"[WARN] {metadata_row}: only "
                #         f"{sum(_ is not None for _ in components.values())} "
                #         f"of 3 components created and saved"
                #     )

                # process waveforms
                step_name = "save_waveforms"  # noqa
                # old_record = dict(record)  # for testing purposes
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
                        clean_record[f] = cast_dtype(val, dtype)
                    except AssertionError:
                        raise AssertionError(f'Invalid value for "{f}": {str(val)}')

                # final checks:
                check_final_metadata(clean_record, h1, h2)

                # finalize clean_record:
                avail_comps, sampling_rate = \
                    finalize_metadata(clean_record, h1, h2, v)
                clean_record['available_components'] = avail_comps
                clean_record['sampling_rate'] = sampling_rate if \
                    pd.notna(sampling_rate) else \
                    int(metadata_fields['sampling_rate']['default'])

                # append metadata:
                new_metadata.append(clean_record)

                # save waveforms
                step_name = "save_waveforms"  # noqa
                file_path = join(dest_waveforms_path, get_file_path(clean_record))
                save_waveforms(file_path, h1, h2, v)

            except Exception as exc:
                logging.error(
                    f"[ERROR] {metadata_row}: {exc} | Step: `{step_name}` | "
                    f"h1_path : {h1_path or 'N/A'} | h2_path : {h2_path or 'N/A'}"
                )
                errs += 1
                continue

            if rec_num / max_rows > (1 - min_waveforms_ok_ratio):
                # we processed enough data (1 - waveforms_ok_ratio)
                ok_ratio = (1 - errs / max_rows)
                if ok_ratio < min_waveforms_ok_ratio:
                    # the processed data error ratio is too high:
                    msg = f'Too many errors ({errs} of {max_rows} records)'
                    print(msg, file=sys.stderr)
                    logging.error(msg)
                    sys.exit(1)

        if new_metadata:
            # save metadata:
            new_metadata_df = pd.DataFrame(new_metadata)
            for col in new_metadata_df.columns:
                new_metadata_df[col] = cast_dtypes(
                    new_metadata_df[col],
                    metadata_fields[col]['dtype']
            )
            new_metadata_df.to_hdf(dest_metadata_path, **hdf_kwargs)

    if isfile(dest_metadata_path):
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


def scan_dir(source_root_dir) -> tuple[dict[str, Union[dict, str]], int]:
    """Scan the given directory"""
    tree = {}
    num_files = 0
    for entry in os.scandir(source_root_dir):
        if entry.is_dir():
            tree[entry.name], _ = scan_dir(entry.path)
            num_files += _
        else:
            file = abspath(entry.path)
            if accept_file(file):
                tree[entry.name] = file
                num_files += 1
    return tree, num_files


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


def read_wafevorms(file_path):
    """Reads the file path previously saved. NOT USED, HERE ONLY FOR REF"""
    with h5py.File(file_path, "r") as f:
        h1_data = f["h1"][:] if "h1" in f else np.array([])
        h1_dt = f["h1"].attrs["dt"] if "h1" in f else None

        h2_data = f["h2"][:] if "h2" in f else np.array([])
        h2_dt = f["h2"].attrs["dt"] if "h2" in f else None

        v_data = f["v"][:] if "v" in f else np.array([])
        v_dt = f["v"].attrs["dt"] if "v" in f else None
    return (h1_dt, h1_data), (h2_dt, h2_data), (v_dt, v_data)


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


def cast_dtypes(values: Any, dtype: Union[str, pd.CategoricalDtype]):
    """
    Cast the values of the output metadata. Safety method (each value in `values` is
    the outcome of `cast_dtype` so it should be of the correct dtype already)
    """
    if dtype == 'int':
        # assert pd.notna(values).all()
        return values.astype(int)
    elif dtype == 'bool':
        return values.astype(bool)
    elif dtype == 'datetime':
        return pd.to_datetime(values, errors='coerce')
    elif dtype == 'str':
        return values.astype(str)
    elif dtype == 'float':
        return values.astype(float)
    elif isinstance(dtype, pd.CategoricalDtype):
        return values.astype(dtype)
    raise AssertionError(f'Unrecognized dtype {dtype}')


def get_file_path(metadata: dict):
    """Return the file (relative) path from the given metadata
    (record metadata already cleaned)"""
    return join(
        str(metadata['event_id']),
        str(metadata['station_id']) + ".h5"
    )


def check_final_metadata(
        metadata: dict,
        h1: Optional[tuple[float, ndarray]],
        h2: Optional[tuple[float, ndarray]]
):

    pga = metadata['PGA']
    pga_c = None
    if h1 is not None and h2 is not None:
        pga_c = np.sqrt(np.max(np.abs(h1[1])) * np.max(np.abs(h2[1])))
    elif h1 is not None and h2 is None:
        pga_c = np.max(np.abs(h1[1]))
    elif h1 is None and h2 is not None:
        pga_c = np.max(np.abs(h2[1]))
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


def finalize_metadata(
        metadata: dict,
        h1: Optional[tuple[float, ndarray]],
        h2: Optional[tuple[float, ndarray]],
        v: Optional[tuple[float, ndarray]]
) -> tuple[Optional[str], Optional[int]]:
    avail_components = (h1 is not None, h2 is not None, v is not None)
    dt = None
    avail_comps = None
    if avail_components == (False, False, True):
        avail_comps = 'V'
        dt = v[0]
    elif avail_components == (True, False, False):
        avail_comps = 'H'
        dt = h1[0]
    elif avail_components == (False, True, False):
        avail_comps = 'H'
        dt = h2[0]
    elif avail_components == (True, True, False):
        avail_comps = 'HH'
        dt = h1[0] if h1[0] == h2[0] else None
    elif avail_components == (True, True, True):
        avail_comps = 'HHV'
        dt = h1[0] if h1[0] == h2[0] == v[0] else None

    sampling_rate = None
    if dt is not None:
        sampling_rate = int(1./dt) if int(1./dt) == float(1./dt) else None
    return avail_comps, sampling_rate


if __name__ == "__main__":
    main()
