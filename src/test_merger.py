import pytest
import merger
import os


def test_check_input_folder_non_existant():
    path = './nonexistant-path'

    with pytest.raises(Exception) as e:
        merger.check_input_folder(folder=path)

    assert e.type == NotADirectoryError

def test_check_output_folder_non_existant(tmpdir):
    path = f'{tmpdir}/subdir'

    merger.check_output_folder(folder=path)

    assert os.path.exists(path=path)

def test_check_output_folder_exists(tmpdir):
    with pytest.raises(Exception) as e:
        merger.check_output_folder(folder=tmpdir)

    assert e.type == OSError
