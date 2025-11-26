import importlib
import importlib.util
import shutil
import os
import sys
from os.path import dirname, join, abspath, isdir, splitext
from io import StringIO, BytesIO
import subprocess
from unittest.mock import patch

# import yaml
# import create_ngawest_dataset, create_kiknet_knet_dataset

import pytest

dest_data_dir = join(abspath(dirname(__file__)), 'tmp')
source_data_dir = join(abspath(dirname(__file__)), 'source_data')


def tearDown():
    shutil.rmtree(dest_data_dir)
    # sys.path.pop(0)


def setUp():
    if not isdir(dest_data_dir):
        os.makedirs(dest_data_dir)
    # src = dirname(dirname(__file__))
    # shutil.copy(join(src, 'create_ngawest_dataset.py'),
    #             join(root, 'create_ngawest_dataset.py'))
    # sys.path.insert(0, root)


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    setUp()
    yield
    tearDown()


def run_(dataset: str):  # ngawest, esm, kinet_knet
    with patch(f"create_{dataset}_dataset.min_waveforms_ok_ratio", 0):

        src_data_dir = join(source_data_dir, dataset)
        metadata = [f for f in os.listdir(src_data_dir) if splitext(f)[1] == '.csv']
        assert len(metadata) == 1, f'CSV in {src_data_dir} must be 1'
        src_metadata_path = join(src_data_dir, metadata[0])

        my_dest_data_dir = join(dest_data_dir, dataset)
        os.mkdir(my_dest_data_dir)
        config_path = join(my_dest_data_dir, "config.yml")

        with open(config_path, 'w') as _:
            _.write(f"""
source_metadata: "{src_metadata_path}"
source_data: "{src_data_dir}"
destination: "{my_dest_data_dir}"
""")
        module_name = f"create_{dataset}_dataset"
        try:
            # project_root = abspath(dirname(dirname(__file__)))
            # os.chdir(project_root)
            # result = subprocess.run(
            #     [sys.executable, f'create_{dataset}_dataset.py', config_path],
            #     capture_output=True,
            #     text=True,
            #     check=True
            # )
            # stdout_text = result.stdout
            # stderr_text = result.stderr
            # asd = 9

            module_path = \
                abspath(join(dirname(dirname(__file__)), f'{module_name}.py'))
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            sys.argv = [f'create_{dataset}_dataset.py', config_path]
            mod.main()
        except SystemExit as err:
            assert err.args[0] == 0

            assert isdir(join(my_dest_data_dir, 'waveforms'))
            # Get printed output
        except Exception as e:
            # Raise a new exception with the subprocess traceback
            raise
        finally:
            del sys.modules[module_name]
        #     # Restore originals
        #     sys.argv = original_argv
        #     sys.stdout = original_stdout


def test_nga_west2():
    run_('ngawest2')


def test_knet():
    run_('kiknet_knet')


def test_esm():
    run_('esm')
