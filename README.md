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

   - Install required packages:
     ```
     pip install --upgrade pip setuptools && pip install obspy pandas pyyaml tqdm
     ```

# Implementation

Create a destination directory where your processed time histories are, say `<DEST>`.

Copy `template.py` and `metadata_fields.yml` in `<DEST>`.

Modify `template.py` (set your Metadata path, and how to read time historeis in the Python file, as well as how to process them, if needed).

**Activate Python virtual environment** and within `DEST`, run:

```
python ./template.py
```

You should see `<DEST>/waveforms` being populated by your processed waveforms (end eventually, a new `<DEST>/metadata.csv` file created therein)




Maintenance:

Generate `metadata_template.csv`. Modify `metadata_fields.yml` and then
```
python -c "import yaml, pandas as pd; _ = open('./metadata_desc.yml'); y=yaml.safe_load(_); _.close();pd.DataFrame(index=list(y.keys()), data=[v['dtype'] for v in y.values()]).T.reset_index(drop=True).to_csv('./metadata_template.csv', index=False)"
```
