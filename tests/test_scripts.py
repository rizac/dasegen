import shutil
import os
import sys
from os.path import dirname, join, abspath, isdir
from io import StringIO
from unittest.mock import patch
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


@patch("create_ngawest_dataset.get_dest_dir_path",
       return_value=join(dest_data_dir, 'nga'))
@patch("create_ngawest_dataset.re_err_th", 1.01)
def test_nga_west2(mocked_dest_dir):
    # Save originals
    original_argv = sys.argv
    original_stdout = sys.stdout

    # try:
    #
    #     # Patch argv and stdout
    #     sys.argv = [
    #         'create_ngawest_dataset.py'
    #     ]
    #     sys.stdout = StringIO()
    #
    #     # Run main
    #
    #     create_ngawest_dataset.main()
    #
    # except SystemExit as err:
    #     assert err.args[0] == 1
    # finally:
    #     # Restore originals
    #     sys.argv = original_argv
    #     sys.stdout = original_stdout

    try:
        # Patch argv and stdout
        sys.argv = [
            'create_ngawest_dataset.py',
            join(source_data_dir, "ngawest2", "ngawest2_metadata.csv"),
            join(source_data_dir, "ngawest2")
        ]
        sys.stdout = StringIO()

        # Run main
        create_ngawest_dataset.main()

    except SystemExit as err:
        assert err.args[0] == 0

        assert isdir(join(mocked_dest_dir.return_value, 'waveforms'))
        # Get printed output
        output = sys.stdout.getvalue()

    finally:
        # Restore originals
        sys.argv = original_argv
        sys.stdout = original_stdout


@patch("create_kiknet_knet_dataset.get_dest_dir_path",
       return_value=join(dest_data_dir, 'knet'))
@patch("create_kiknet_knet_dataset.re_err_th", 1.01)
def test_knet(mocked_dest_dir):
    # Save originals
    original_argv = sys.argv
    original_stdout = sys.stdout

    try:

        # Patch argv and stdout
        sys.argv = [
            'create_kiknet_knet_dataset.py'
        ]
        sys.stdout = StringIO()

        # Run main

        create_kiknet_knet_dataset.main()

    except SystemExit as err:
        assert err.args[0] == 1
    finally:
        # Restore originals
        sys.argv = original_argv
        sys.stdout = original_stdout

    try:
        # Patch argv and stdout
        sys.argv = [
            'create_kiknet_knet_dataset.py',
            join(source_data_dir, "kiknet_knet", "kiknet_knet_metadata.csv"),
            join(source_data_dir, "kiknet_knet")
        ]
        sys.stdout = StringIO()

        # Run main

        create_kiknet_knet_dataset.main()

    except SystemExit as err:
        assert err.args[0] == 0

        assert isdir(join(mocked_dest_dir.return_value, 'waveforms'))
        # Get printed output
        output = sys.stdout.getvalue()

    finally:
        # Restore originals
        sys.argv = original_argv
        sys.stdout = original_stdout
