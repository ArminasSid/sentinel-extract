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

    if not os.path.exists(args.input_folder):
        raise NotADirectoryError('Input folder not found.')

    path = '{}'.format(args.output_folder)
    if not os.path.exists(path):
        os.makedirs(path)

    return args

def BuildVRT(output_path, output_name, input_paths):
    output = "{}/{}".format(output_path, "{}.vrt".format(output_name))
    gdal.SetConfigOption('GDAL_NUM_THREADS', '4')
    vrt = gdal.BuildVRT(output, input_paths)
    print("Vrt file {}.vrt created.".format(output_name))
    return vrt

def Merge(vrt, output_path, output_name):
    # vrt = "{}/{}".format(output_path, "{}.vrt".format(output_name))
    gdal.SetConfigOption('GDAL_VRT_ENABLE_PYTHON', 'YES')
    gdal.SetConfigOption('GDAL_CACHEMAX', '1024')
    opts = gdal.TranslateOptions(format='GTIFF', creationOptions="NUM_THREADS=ALL_CPUS")
    gdal.Translate("{}/{}".format(output_path, "{}.tiff".format(output_name)), vrt, options=opts)

    print("Translate complete. File {}.tiff created.".format(output_name))

def Merge_Images(input_folder, output_folder, output_name):
    images = glob('{}/*.jp2'.format(input_folder))
    if len(images) == 0:
        images = glob('{}/*.tiff'.format(input_folder))

    vrt = BuildVRT(output_folder, output_name, images)

    Merge(vrt, output_folder, output_name)



if __name__=='__main__':
    args = parse_args()

    Merge_Images(args.input_folder, args.output_folder, args.output_name)
