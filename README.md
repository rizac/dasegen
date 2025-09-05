This package can be downloaded and modified for generating datasets of processed time histories (accelerometers, velocimeters or displacement).

# Requirements


1. Raw time histories, where each time history a waveform-like files denoting one of the 3 components of a recorded earthquake
   (two horizontal, `h1` and `h2`, one vertical, `v`). Common formats might be 'KNET', 'MSEED' or 'SAC'.
   
3. Metadata associated to each waveform. Metadata can be stored as hsd or csv and each record (row) will describe the metadata of a recorded earthquake

4. The Python code templates and libraries.

   - Clone this package

   - Install a Python virtual environment:
     ```
     python3 -m venv .env 
     ```

   - Activate the environment **to be done every time you run Python code**
     ```
     source .env/bin/activate
     ```

   - Install required packages (add any package of your choice if needed):
     ```
     pip install --upgrade pip setuptools && pip install pandas h5py pyyaml tqdm
     ```
    <!-- pip install --upgrade pip setuptools && pip install obspy pandas pyyaml tqdm -->
# Implementation


1. Copy `create_dataset.py` as well as `metadata_fields.yml` in a empty directory
   
2. Edit your source metadata file (CSV format) to match the field names in `metadata_fields.yml`.
   You can also start from `metadata_template.csv` as empty template, leaving empty cells if
   data is N/A or missing, or you plan to fill it inside `create_dataset.py` 

3. Edit `create_dataset.py`

   3a. Set the path of the source metadata file (variable `source_metadata_path`)

   3b. Implement how to read time histories from the metadata file rows
       (functions `get_waveforms_path` and `read_waveform`)

   3c. Implement how to process time histories and potentially modify the associated CSV row
       (function `process_waveforms`)

4. Eventually, execute `create_dataset.py` file *on the terminal within the Python virtual environment*
   (or Conda env):
   ```
   python3 create_dataset.py
   ```
   The file will scan all rows of your source metadata file, process them and put them in the
   `waveforms` sub-directory of the root directory of `create_dataset.py`. A new metadata file `metadata.csv`
   will be also created in the same directory



# Tips

## K-NET and KIK-NET time histories

### Opening K-Net / Kik-NET

To open K-Net /Kik-Net time histories, you can use obspy
```python
file_path = "..."  # your path here
stream = obspy.read(file_path, format='KNET')
trace = stream[0]
```


Your data (`numpy.array`) is in the trace.data attribute:
```python
print(trace.data)
```
```python
array([8009., 9006., 7998., ..., 8005., 8009., 8005.])
```


Your metadata is in the trace.stats attribute:
```python
print(trace.stats)
```
```python
network: BO
station: MYG002
location: 
channel: NS
starttime: 2006-01-18T14:25:31.000000Z
endtime: 2006-01-18T14:25:31.470000Z
sampling_rate: 100.0
delta: 0.01
npts: 48
calib: 6.340209495401812e-06
_format: KNET
knet: AttribDict({'evot': UTCDateTime(2006, 1, 18, 14, 25), 'evla': 37.798, 'evlo': 142.2, 'evdp': 36.0, 'mag': 5.7, 'stla': 38.7262, 'stlo': 141.5109, 'stel': 79.0, 'duration': 108.0, 'accmax': 48.06, 'last correction': UTCDateTime(2006, 1, 18, 14, 25, 31)})
```


Obspy adds two custom attributes, the most important is `trace.stats.knet`, where 
you can access K-NEt specific metadata:
```python
print(trace.stats.knet.evot)
```
```python
UTCDateTime(2006, 1, 18, 14, 25)
```

### Saving K-Net / Kik-NET as miniSEED

To save file as miniSeed:
```python
file_path = "..."  # your path here
Stream[trace]).write(file_path, format='MSEED')
```

**NOTE** You will not be able to save to file all custom attributes (i.e., `trace.stats.knet`), **If you need to save them, you will need to add new metadata fields in the associated CSV** (previous discussion needed probably)

<!--

Generate `metadata_template.csv`. Modify `metadata_fields.yml` if needed, activate virtual environment and then
```
python -c "import yaml, pandas as pd; _ = open('./metadata_fields.yml'); y=yaml.safe_load(_); _.close();pd.DataFrame(index=list(y.keys())).T.reset_index(drop=True).to_csv('./metadata_template.csv', index=False)"
```

-->
