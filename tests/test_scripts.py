import shutil
import os
import sys
from os.path import dirname, join, abspath, isdir
from io import StringIO, BytesIO
from unittest.mock import patch

import yaml

import create_ngawest_dataset, create_kiknet_knet_dataset

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


@patch("create_ngawest_dataset.waveforms_ok_ratio", 1.01)
def test_nga_west2():

    my_dest_data_dir = join(dest_data_dir, 'nga')
    os.mkdir(my_dest_data_dir)
    config_path = join(my_dest_data_dir, "config.yml")

    with open(config_path, 'w') as _:
        _.write(f"""
source_metadata: "{join(source_data_dir, "ngawest2", "ngawest2_metadata.csv")}"
source_data: "{join(source_data_dir, "ngawest2")}"
destination: "{my_dest_data_dir}"
""")

    # Save originals
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        # Patch argv and stdout
        sys.argv = [
            'create_ngawest_dataset.py',
            config_path
        ]
        sys.stdout = StringIO()

        # Run main
        create_ngawest_dataset.main()

    except SystemExit as err:
        assert err.args[0] == 0

        assert isdir(join(my_dest_data_dir, 'waveforms'))
        # Get printed output
        output = sys.stdout.getvalue()

    finally:
        # Restore originals
        sys.argv = original_argv
        sys.stdout = original_stdout


@patch("create_kiknet_knet_dataset.waveforms_ok_ratio", 1.01)
def test_knet():
    my_dest_data_dir = join(dest_data_dir, 'knet')
    os.mkdir(my_dest_data_dir)
    config_path = join(my_dest_data_dir, "config.yml")

    with open(config_path, 'w') as _:
        _.write(f"""
source_metadata: "{join(source_data_dir, "kiknet_knet", "kiknet_knet_metadata.csv")}"
source_data: "{join(source_data_dir, "kiknet_knet")}"
destination: "{my_dest_data_dir}"
""")

    # Save originals
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:
        # Patch argv and stdout
        sys.argv = [
            'create_kiknet_knet_dataset.py',
            config_path
        ]
        sys.stdout = StringIO()

        # Run main
        create_kiknet_knet_dataset.main()

    except SystemExit as err:
        assert err.args[0] == 0

        assert isdir(join(my_dest_data_dir, 'waveforms'))
        # Get printed output
        output = sys.stdout.getvalue()

    finally:
        # Restore originals
        sys.argv = original_argv
        sys.stdout = original_stdout
