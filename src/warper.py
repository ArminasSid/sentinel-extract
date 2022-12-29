import os
from osgeo import gdal
import argparse
import shutil
from tqdm import tqdm
import numpy as np
import warnings
from glob import glob
from dataclasses import dataclass

warnings.filterwarnings("ignore")


@dataclass
class Coordinates:
    xmin: float
    ymax: float
    xmax: float
    ymin: float


@dataclass
class Product:
    custom_raster_path: str
    tci_raster_path: str
    mask_raster_path: str

    custom_raster: gdal.Dataset = None
    tci_raster: gdal.Dataset = None
    mask_raster: gdal.Dataset = None

    coordinates: Coordinates = None

    def __post_init__(self):
        self.custom_raster = gdal.Open(self.custom_raster_path)
        self.tci_raster = gdal.Open(self.tci_raster_path)
        self.mask_raster = gdal.Open(self.mask_raster_path)

        self.coordinates = Coordinates(
            self.get_raster_coordinates(raster=self.tci_raster))

    @staticmethod
    def get_raster_coordinates(raster):
        """
        Get raster coordinates from gdal.Dataset object
        """
        gt = raster.GetGeoTransform()
        xmin = gt[0]
        ymax = gt[3]
        res = gt[1]
        xlen = res * raster.RasterXSize
        ylen = res * raster.RasterYSize
        xmax = xmin + xlen
        ymin = ymax - ylen
        return [xmin, ymax, xmax, ymin]

    def fits_coord_req(self, w_x_y) -> bool:
        """
        Checks whether product is a super set of wanted x and y coordinates (w_x_y)

        w_x_y - Wanted coordinate bounds
        """
        if (self.coordinates.xmin < w_x_y[0]
                and self.coordinates.ymax > w_x_y[1]
                and self.coordinates.xmax > w_x_y[2]
                and self.coordinates.ymin < w_x_y[3]):
            return True
        return False


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

    check_if_is_directory(folder_path=args.input_folder)
    process_output_folder(
        output_path=args.output_folder,
        overwrite_allowed=args.overwrite
    )

    return args


def check_if_is_directory(folder_path: str):
    if not os.path.exists(path=folder_path):
        raise NotADirectoryError('Input folder not found.')


def process_output_folder(output_path: str, overwrite_allowed: bool):
    if not os.path.exists(path=output_path):
        # If folder doesn't exist, just create it
        os.makedirs(name=output_path)
    elif not overwrite_allowed:
        # Folder exists, overwrite not enabled
        raise PermissionError(
            'Folder exists, add --overwrite if you want to overwrite it.'
        )
    else:
        # Folder exists, overwrite enabled
        shutil.rmtree(path=output_path)
        os.makedirs(name=output_path)


def tile_coords(tile):
    if str(tile).lower() == 'south':
        return Coordinates(
            xmin=22.620849609375, ymax=54.983918190363234,
            xmax=26.015625000000, ymin=53.878440403328830
        )
    elif str(tile).lower() == 'north':
        return Coordinates(
            xmin=20.84111110000, ymax=56.46855975598276,
            xmax=26.87255859375, ymin=54.89556479077338
        )
    else:
        raise KeyError('Tile name not found.')


def image_coordinates(img):
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
    if np.count_nonzero(data == np.zeros(3, dtype=int)) > 10:
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

def calculate_iterations1(bounds, step):
    


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


def filter_images1(custom_rasters, tci_rasters, mask_rasters, coords_rasters, w_x_y):
    """
    Filter rasters by wanted coordinates (w_x_y).
    """
    custom_rasters_fltr, tci_rasters_fltr = []
    mask_rasters_fltr, coords_rasters_fltr = []

    # Filter rasters that fit needed coords
    zipped = zip(custom_rasters, tci_rasters, mask_rasters, coords_rasters)
    for custom, tci, mask, coord in zipped:
        if check_if_fits_req(coord, w_x_y):
            custom_rasters_fltr.append(custom)
            tci_rasters_fltr.append(tci)
            mask_rasters_fltr.append(mask)
            coords_rasters_fltr.append(coord)
    return custom_rasters_fltr, tci_rasters_fltr, mask_rasters_fltr, coords_rasters_fltr


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


def cut_normal1(tile, warping_bounds, products, output_folder, step):
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


def get_rasters(input_folder: str):
    fci_rasters = glob(pathname=f'{input_folder}/*FCI*.tiff')
    tci_rasters = glob(pathname=f'{input_folder}/*TCI*.tiff')
    mask_rasters = glob(pathname=f'{input_folder}/*mask*.tiff')

    return fci_rasters, tci_rasters, mask_rasters


def warp(input_folder, output_folder):
    files = os.listdir(input_folder)

    fci_rasters, tci_rasters, mask_rasters = get_rasters(
        input_folder=input_folder)

    products = []
    # Assemble products into a list of Product objects
    for fci_raster, tci_raster, mask_raster in zip(fci_rasters, tci_rasters, mask_rasters):
        products.append(Product(
            custom_raster_path=fci_raster,
            tci_raster_path=tci_raster,
            mask_raster_path=mask_raster
        ))

    print("Detected {} images, loading images as rasters...".format(len(products)))

    tiles = ["north", "south"]

    rgb = is_image_rgb(tci_rasters[0])

    step = 0.00642
    for tile in tiles:
        print("Forming {} part".format(tile))
        bounds = tile_coords(tile=tile)
        cut_normal(
            tile=tile,

        )
        cut_normal(tile, ul_x_y, lr_x_y, images,
                   masks, output_folder, step, rgb)


def main():
    args = parse_args()
    warp(args.input_folder, args.output_folder)


if __name__ == '__main__':
    main()
