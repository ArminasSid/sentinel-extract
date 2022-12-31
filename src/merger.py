from osgeo import gdal
import argparse
import os
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Image merger.')
    # model and dataset
    parser.add_argument("-i", '--input_folder', type=str, default='cuts',
                        help='input folder path (default: cuts)')
    parser.add_argument("-o", '--output_folder', type=str, default='merged',
                        help='output folder path (default: merged)')
    parser.add_argument("-n", '--output_name', type=str, default='image',
                        help='Output name. (default: image')

    args = parser.parse_args()

    check_input_folder(folder=args.input_folder)
    check_output_folder(folder=args.output_folder)

    return args


def check_input_folder(folder):
    if not os.path.exists(path=folder):
        raise NotADirectoryError('Input folder not found.')

def check_output_folder(folder):
    if not os.path.exists(path=folder):
        os.makedirs(name=folder)
    else:
        raise OSError('Folder already exists.')

def merge(input_folder: str, output_folder: str, output_name: str):
    # Get list of rasters
    images = glob(f'{input_folder}/*.tiff')

    print(f'Building vrt file for: {output_name}')
    vrt = gdal.BuildVRT(destName='', srcDSOrSrcDSTab=images)

    print(f'Building complete mosaic raster: {output_name}')
    gdal.Translate(
        destName=f'{output_folder}/{output_name}',
        srcDS=vrt,
        options=gdal.TranslateOptions(
            creationOptions='NUM_THREADS=ALL_CPUS',
            format='GTIFF'
        )   
    )


if __name__=='__main__':
    args = parse_args()

    merge(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        output_name=args.output_name
    )
