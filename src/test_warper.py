import pytest
import warper
from osgeo import gdal
import numpy as np
import os


# Test warper functions

def test_check_if_directory_exists_raise():
    with pytest.raises(Exception) as e:
        warper.check_if_is_directory('./not_a_dir')

    assert e.type == NotADirectoryError


def test_process_output_folder_no_folder(tmp_path):
    path = f'{tmp_path}/subdir'

    warper.process_output_folder(output_path=path, overwrite_allowed=False)

    assert os.path.exists(path)

def test_process_output_folder_exists_folder_no_overwrite(tmp_path):
    with pytest.raises(Exception) as e:
        warper.process_output_folder(output_path=tmp_path, overwrite_allowed=False)

    assert e.type == PermissionError


def test_process_output_folder_exists_folder_enabled_overwrite(tmp_path):
    path = f'{tmp_path}/subdir'
    os.makedirs(name=path)

    warper.process_output_folder(output_path=tmp_path, overwrite_allowed=True)

    assert not os.path.exists(path=path)
    assert os.path.exists(path=tmp_path)


def test_is_img_rgb_positive(mocker):
    # gdal.Dataset
    mocker.patch('osgeo.gdal.Dataset.ReadAsArray', return_value=np.zeros((10, 10, 10)))

    arr = np.zeros((10, 10, 3))
    print(arr.ndim)

    assert warper.is_image_rgb(gdal.Dataset) == True

def test_is_img_rgb_negative(mocker):
    # gdal.Dataset
    mocker.patch('osgeo.gdal.Dataset.ReadAsArray', return_value=np.zeros((10)))

    arr = np.zeros((10, 10, 1))
    print(arr.ndim)

    assert warper.is_image_rgb(gdal.Dataset) == False


def test_contains_dead_pixels_positive(mocker):
    mocker.patch('osgeo.gdal.Dataset.ReadAsArray', return_value=np.ones((100)))

    assert not warper.contains_dead_pixels(gdal.Dataset)

def test_contains_dead_pixels_positive(mocker):
    mocker.patch('osgeo.gdal.Dataset.ReadAsArray', return_value=np.zeros((100)))

    assert warper.contains_dead_pixels(gdal.Dataset)


def test_contains_dead_pixels_rgb_negative(mocker):
    mocker.patch('osgeo.gdal.Dataset.ReadAsArray', return_value=np.ones((100, 100, 100)))

    assert not warper.contains_dead_pixels_rgb(gdal.Dataset)

def test_contains_dead_pixels_rgb_positive(mocker):
    mocker.patch('osgeo.gdal.Dataset.ReadAsArray', return_value=np.zeros((3, 100, 100)))

    assert warper.contains_dead_pixels_rgb(gdal.Dataset)


def test_calculate_iterations_startpoint_greater_than_endpoint():
    result = warper.calculate_iterations(
        starting_point=100,
        end_point=50,
        step=10
    )

    assert result > 0

def test_calculate_iterations_endpoint_greater_than_startpoint():
    result = warper.calculate_iterations(
        starting_point=50,
        end_point=100,
        step=10
    )
    
    assert result > 0


def test_load_rasters_3(mocker):
    mocker.patch('osgeo.gdal.Open', return_value=np.zeros(10, dtype=int))
    raster_paths = [
        'path1',
        'path2',
        'path3'
    ]

    result = warper.load_rasters(image_paths=raster_paths)

    assert len(result) == 3
    assert np.all((result == np.zeros(10, dtype=int)))
