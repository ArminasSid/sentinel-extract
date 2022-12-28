import argparse
import os
import shutil
from src import converter, warper, merger
from src import extracter_v1 as extracter
import pandas as pd
from glob import glob


def parse_args():
    parser = argparse.ArgumentParser(description='Process Sentinel-2 data into a singular image of area of interest.')
    # model and dataset
    parser.add_argument('-c', '--configuration', type=str, default='input.csv',
                        help='Input csv data (Default: input.csv)')
    parser.add_argument("-i", '--input_folder', type=str, default='input_folder',
                        help='Input folder with rasters (Default: input_folder)')
    parser.add_argument("-o", '--output_folder', type=str, default='output_folder',
                        help='Output folder path (Default: output_folder)')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite output folder if exists')

    args = parser.parse_args()
    return args
    

def validate_args(args) -> None:    
    # Check if inputs exist
    if not os.path.isdir(args.input_folder):
        raise NotADirectoryError(f'Input folder: {args.input_folder} was not found.')
    if not os.path.isfile(args.configuration):
        raise FileNotFoundError(f'Input configuration file: {args.configuration} was not found.')

    # Check if output already exists
    if os.path.exists(args.output_folder):
        # Now that output_folder exists, check if we can delete it
        if args.overwrite:
            shutil.rmtree(args.output_folder)
        else:
            # Cannot delete, raise permission error
            raise PermissionError(f'Output directory exists. However --overwrite permission was not provided.')

            
def do_extracting(args):
    # Basic extraction of zip files (Hardcoded to use raster masks instead of polygon masks) 
    output = f'{args.output_folder}/extracted'
    os.makedirs(name=output, exist_ok=True)
    
    # systems_of_interest = ['T34', 'T35']
    # folders = []
    # for element in systems_of_interest:
    #     # Create folder path
    #     folders.append(f'{output}/{element}')
        
    # for folder in folders:
    #     # Create folders if they don't exist yet
    #     os.makedirs(name=folder, exist_ok=True)
    
    # Read csv, sort by cloud cover and ingestion date
    products_df = pd.read_csv(args.configuration, index_col=0)
    products_df_sorted = products_df.sort_values(
        ['cloudcoverpercentage', 'ingestiondate'], 
        ascending=[True, True])
    
    # Iterate over every product and uzip useful parts
    zipfiles = products_df_sorted['title'].tolist()
    # Extract products
    extracter.extract(
        input_folder=args.input_folder,
        input_zipfiles=zipfiles,
        output_folder=output
    )
    # for product in tqdm(products_df_sorted['title'].tolist()):
    #     # Extract products
    #     extracter.extract(
    #         input_folder=args.input_folder,
    #         input_zipfiles=
    #     )

        
        # Hardcoded TCI value (True Color Image) 
        # extracter.pull_products0(args.input_folder, 'TCI', '{}.zip'.format(product), folders)
    
    return output    

        
def do_converting(args, extracted):
    # Define output folder for reprojected images
    output = f'{args.output_folder}/reprojected'
    os.makedirs(name=output, exist_ok=True)
    
    # Get all input folders
    input_folders = glob(f'{extracted}/*')
    
    # Reprojection
    for folder in input_folders:
        converter.formatter(input_folder=folder, output_folder=output)
        
    return output


def do_warping(args, reprojected):
    # Define output folder for warped rasters
    output = f'{args.output_folder}/warped'
    os.makedirs(name=output, exist_ok=True)
    
    # Warp images
    warper.warp(
        input_folder=reprojected,
        output_folder=output
    )
    
    return output


def do_merging(args, warped):
    # Create output folder for merged raster
    output = f'{args.output_folder}/results'
    os.makedirs(name=output, exist_ok=True)
    
    # Define output name
    name = 'raster'
    
    # Merge images
    merger.Merge_Images(
        input_folder=warped,
        output_folder=output,
        output_name=name
    )
    
    return output
        

def main():
    # Get terminal arguments
    args = parse_args()
    
    # Validate provided data
    validate_args(args=args)
    
    # Create output_folder
    os.makedirs(name=args.output_folder)
    
    # Begin processing
    # Extract images
    # ------
    print('Initiating image extraction.')
    output_extracted = do_extracting(args=args)
    # ------
    
    # Reporoject images (Change coordinate systems to WGS84)
    # ------
    # print('Initiating image reprojection.')
    # output_reprojected = do_converting(args=args, extracted=output_extracted)
    # ------
    
    # Warp images into a mosaic, while removing clouds
    # ------
    print('Initiating image warping.')
    output_warped = do_warping(args=args, reprojected=output_extracted)
    # ------
    
    # Merge images into final single image
    # ------
    print('Initiating image merging')
    output_merged = do_merging(args=args, warped=output_warped)
    # ------    
    

if __name__=='__main__':
    main()
