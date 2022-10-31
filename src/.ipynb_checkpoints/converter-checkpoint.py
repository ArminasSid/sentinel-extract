from osgeo import gdal
from glob import glob
import sys, os
import argparse
import shutil
from tqdm import tqdm

def parse_args():
    """Reprojection options"""
    parser = argparse.ArgumentParser(description='Reprojection with gdal.')
    # model and dataset
    parser.add_argument('--input_folder', type=str, default='T34',
                        help='Input folder (default: T34)')
    parser.add_argument('--output_folder', type=str, default='Reprojected',
                        help='output_folder (default: Reprojected)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite output dir if exists, removes all items in existing output directory.')
    parser.add_argument('--add', action='store_true', default=False,
                        help='Add to output dir if exists, adds to same directory, if there is a conflict overwrites.')                    

    args = parser.parse_args()


    if not args.overwrite and not args.add:
        raise PermissionError('Adding or overwriting not enabled. Run command with --overwrite or --add.')
    if args.overwrite:
        print('Output folder {} exists. Overwrite enabled. Removing all items in directory...'.format(args.output_folder))
        shutil.rmtree(args.output_folder)
        print('Removed all items in directory {}.'.format(args.output_folder))

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        print('Created {} directory.'.format(args.output_folder))

    return args

def ChangeFormat(_input_folder, _output_folder, _input_epsg, _output_epsg):
    Files = os.listdir(_input_folder)
    print('Reprojecting {} directory.'.format(_input_folder))
    input_epsg = 'EPSG:{}'.format(_input_epsg)
    output_epsg = 'EPSG:{}'.format(_output_epsg)
    # tbar = tqdm(Files)
    for file in tqdm(Files):
        input_path = "{}/{}".format(_input_folder, file)
        output_path = "{}/{}".format(_output_folder, file)
        rePrj = gdal.Warp(srcDSOrSrcDSTab=input_path,
                    destNameOrDestDS=output_path,
                    srcSRS=input_epsg,
                    format='JP2OpenJPEG',
                    warpOptions = "NUM_THREADS=ALL_CPUS",
                    dstSRS=output_epsg)
        # print(gdal.Info(rePrj))

def formatter(input_folder, output_folder):
    # Additional formats, for specific tile folders may be added here.
    folder_type = input_folder[input_folder.rfind('/')+1:]
    print(f'Folder type detected: {folder_type}')
    if folder_type == 'T34':
        ChangeFormat(input_folder, output_folder, '32634', '4126')
    elif folder_type == 'T35':
        ChangeFormat(input_folder, output_folder, '32635', '4126')


if __name__=='__main__':
    args = parse_args()
    input_folders = glob("{}/*".format(args.input_folder))
    for folder in input_folders:
        formatter(folder, args.output_folder)
    
