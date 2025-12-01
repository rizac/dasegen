import sys
import os
from os.path import abspath, join, isdir
import glob
import re
import json
import pandas as pd
from tqdm import tqdm  # pip install tqdm

# --- 1) check arguments ---
if len(sys.argv) != 3:
    print("Usage: python build_report.py <source_dataset_directory> "
          "<destination_directory>")
    sys.exit(1)

base_dir = abspath(sys.argv[1])
dest_dir = abspath(sys.argv[2])
for _ in [base_dir, dest_dir]:
    if not isdir(_):
        print("Error: directory does not exist")
        sys.exit(1)

# --- 2) scan all *metadata*.hdf files ---
hdf_files = glob.glob(join(base_dir, "**", "*metadata*.hdf"), recursive=True)
if not hdf_files:
    print("No metadata HDF files found in", base_dir)
    sys.exit(0)

# --- 3) helper functions for formatting ---
def fmt_coord(x):
    x3 = round(x, 3)
    return int(x3) if x3.is_integer() else x3


def fmt_mag(m):
    m1 = round(m, 1)
    return int(m1) if m1.is_integer() else m1


# --- 4) initialize data object ---
data_js = {}

# --- 5) precompute number of events per dataset ---
event_counts = {}
total_events = 0
for fpath in hdf_files:
    try:
        df_ev = pd.read_hdf(fpath, columns=['event_id'])
        n_events = df_ev['event_id'].nunique()
        dataset_name = os.path.basename(os.path.dirname(fpath))
        event_counts[fpath] = n_events
        total_events += n_events
    except Exception as e:
        print(f"Error reading {fpath} for counting events: {e}")
        event_counts[fpath] = 0

# --- 6) process files with tqdm progress bar ---
with tqdm(total=total_events, unit="event") as pbar:
    for fpath in hdf_files:
        dataset_name = os.path.basename(os.path.dirname(fpath))
        data_js.setdefault(dataset_name, [])

        try:
            cols = [
                'event_id', 'station_id', 'magnitude', 'event_latitude',
                'event_longitude', 'station_latitude', 'station_longitude',
                'available_components'
            ]
            df: pd.DataFrame = pd.read_hdf(fpath, columns=cols)  # noqa

            # group by event_id, preserve order
            grouped_events = df.groupby('event_id', sort=False)

            for eid, event_group in grouped_events:
                ev_row = event_group.iloc[0]
                evlat = fmt_coord(ev_row['event_latitude'])
                evlon = fmt_coord(ev_row['event_longitude'])
                magnitude = fmt_mag(ev_row['magnitude'])
                event_array = [evlat, evlon, magnitude]

                # group stations within event by station_id
                for _, st_group in event_group.groupby('station_id', sort=False):
                    st_row = st_group.iloc[0]
                    stlat = fmt_coord(st_row['station_latitude'])
                    stlon = fmt_coord(st_row['station_longitude'])
                    n_waves = st_group['available_components'].str.len().sum()
                    event_array.extend([stlat, stlon, n_waves])

                data_js[dataset_name].append(event_array)
                pbar.update(1)  # increment per event

        except Exception as e:
            print(f"\nError reading {fpath}: {e}")
            continue

# --- 7) convert to compact JSON ---
data_str = json.dumps(data_js, separators=(',', ':'))

# --- 8) read HTML template ---
html_template_path = os.path.join(
    os.path.dirname(__file__), "dasegen-stats-template.html"
)
if not os.path.isfile(html_template_path):
    print("Error: template.html not found")
    sys.exit(1)

with open(html_template_path, 'r') as f:
    html_content = f.read()

# replace const data
html_content_new = re.sub(r'const\s+data\s*=\s*\{.*?\};',
                          f'const data = {data_str};',
                          html_content,
                          flags=re.DOTALL)

# --- 9) write output ---
output_path = os.path.join(dest_dir, "dasegen-stats.html")
with open(output_path, 'w') as f:
    f.write(html_content_new)

print(f"Report written to {output_path}")
