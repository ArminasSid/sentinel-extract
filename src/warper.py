import os
from PIL import Image
from osgeo import gdal, ogr
import pandas as pd
import argparse
import shutil
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Warper, if input folder contains kml masks, will use them.')
    # model and dataset
    parser.add_argument("-i", '--input_folder', type=str, default='2020/LKS94_Reprojected',
                        help='input folder path (default: LKS94_Reprojected)')
    parser.add_argument("-o", '--output_folder', type=str, default='Cuts',
                        help='output folder path (default: Cuts)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite output folder if exists')

    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        raise NotADirectoryError('Input folder not found.')

    path = '{}'.format(args.output_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    elif not args.overwrite:
        raise PermissionError(
            'Folder exists, add --overwrite if you want to overwrite it.')
    else:
        shutil.rmtree(path)
        os.makedirs(path)

    return args


def tile_coords(tile):
    if str(tile).lower() == 'south':
        ul_x_y = [22.620849609375, 54.983918190363234]
        lr_x_y = [26.015625, 53.87844040332883]
    elif str(tile).lower() == 'north':
        ul_x_y = [20.91796875, 56.46855975598276]
        lr_x_y = [26.87255859375, 54.895564790773385]
    else:
        raise KeyError('Tile name not found.')
    return ul_x_y, lr_x_y


def image_coordinates(img):
    #img = gdal.Open(image)
    gt = img.GetGeoTransform()
    xmin = gt[0]
    ymax = gt[3]
    res = gt[1]
    xlen = res * img.RasterXSize
    ylen = res * img.RasterYSize
    xmax = xmin + xlen
    ymin = ymax - ylen
    return [xmin, ymax, xmax, ymin]


def contains_dead_pixels(image):
    data = image.ReadAsArray()
    if np.count_nonzero(data == np.zeros(1, dtype=int)) > 10:
        return True
    return False


def contains_dead_pixels_rgb(image):
    data = np.transpose(image.ReadAsArray(), axes=(1, 2, 0))
    if np.count_nonzero(data == np.zeros((3), dtype=int)) > 10:
        return True
    return False


def contains_clouds(image):
    data = image.ReadAsArray().flatten()
    if np.count_nonzero(data > 40) > 120:
        return True
    return False


def check_if_fits_req(x_y, w_x_y):
    if not (x_y[0] < w_x_y[0]
            and x_y[1] > w_x_y[1]
            and x_y[2] > w_x_y[2]
            and x_y[3] < w_x_y[3]):
        return False
    return True


def calculate_iterations(starting_point, end_point, step):
    if starting_point > end_point:
        return ((starting_point - end_point) / step)
    else:
        return ((end_point - starting_point) / step)


def wanted_x_y(ul_x_y, step, i_offset, j_offset):
    return [
        ul_x_y[0] + (i_offset * step),
        ul_x_y[1] - (j_offset * step),
        ul_x_y[0] + ((i_offset + 1) * step + step),
        ul_x_y[1] - ((j_offset + 1) * step + step)
    ]


def filter_images(images, masks, coords, w_x_y):
    filt_img = []
    filt_msk = []
    filt_coords = []
    # Filter images that fit needed coords
    for image, mask, coord in zip(images, masks, coords):
        if check_if_fits_req(coord, w_x_y):
            filt_img.append(image)
            filt_msk.append(mask)
            filt_coords.append(coord)
    return filt_img, filt_msk, filt_coords


def warp_memory(image, bounds):
    return gdal.Warp('', image, format='VRT', outputBounds=bounds)


def warp_to_file(image, output_path, bounds):
    return gdal.Warp(output_path, image, format='GTiff', outputBounds=bounds)


def produce_image_without_clouds(bounds, output_path, images, masks, rgb):
    fit = None
    for image, mask in zip(images, masks):
        img = warp_memory(image, bounds)
        # Check for dead pixels
        if rgb:
            if contains_dead_pixels_rgb(img):
                continue
        else:
            if contains_dead_pixels(img):
                continue
        if fit is None:
            fit = image
        msk = warp_memory(mask, bounds)
        # Check for clouds
        if contains_clouds(msk):
            continue

        warp_to_file(image, output_path, bounds)
        return True

    if fit is not None:
        warp_to_file(fit, output_path, bounds)
        return True
    return False


def cut_normal(tile, ul_x_y, lr_x_y, _input_images, _input_masks, output_folder, step, rgb):
    i_range = range(int(calculate_iterations(ul_x_y[0], lr_x_y[0], step)))
    j_range = range(int(calculate_iterations(ul_x_y[1], lr_x_y[1], step)))
    _input_coords = []
    for image in _input_images:
        _input_coords.append(image_coordinates(image))
    tbar = tqdm(i_range)
    for i in tbar:
        for j in j_range:
            w_x_y = wanted_x_y(ul_x_y, step, i, j)

            filtered_images, filtered_masks, filtered_coords = filter_images(
                _input_images,
                _input_masks,
                _input_coords,
                w_x_y
            )
            number = '{0}_{1}'.format(str(i).zfill(5), str(j).zfill(5))
            output_image = "{}/image_{}_{}.jp2".format(
                output_folder, tile, number)
            bounds = [w_x_y[0], w_x_y[3], w_x_y[2], w_x_y[1]]

            produce_image_without_clouds(
                bounds, output_image, filtered_images, filtered_masks, rgb)


def load_rasters(image_paths):
    rasters = []
    for path in image_paths:
        rasters.append(gdal.Open(path))
    return rasters


def is_image_rgb(image):
    arr = image.ReadAsArray()
    if arr.ndim == 3:
        return True
    else:
        return False


def warp(input_folder, output_folder):
    files = os.listdir(input_folder)
    input_images = []
    input_masks = []
    for i in range(int(len(files)/2)):
        input_images.append('{}/image_{}.jp2'.format(input_folder, i))
        input_masks.append('{}/mask_{}.jp2'.format(input_folder, i))
    print("Detected {} images, loading images as rasters...".format(len(input_images)))

    images = load_rasters(input_images)
    masks = load_rasters(input_masks)
    tiles = ["north", "south"]
    print("Images and masks have been loaded in. Proceding to warping.")

    rgb = is_image_rgb(images[0])

    step = 0.00642
    for tile in tiles:
        print("Forming {} part".format(tile))
        ul_x_y, lr_x_y = tile_coords(tile)
        cut_normal(tile, ul_x_y, lr_x_y, images,
                   masks, output_folder, step, rgb)


def main():
    args = parse_args()
    warp(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()
