from osgeo import gdal, gdalconst
from zipfile import ZipFile
import fnmatch
from typing import List
import tempfile
import pandas as pd
from tqdm import tqdm
from glob import glob
import os
import shutil


def find_band(file_list: List[str], band: str) -> str:
    # Hardcoded Sentinel-2 precision values
    precisions = ['R10m', 'R20m', 'R60m']

    for precision in precisions:
        files = fnmatch.filter(names=file_list, pat=f'**/{precision}/**{band}**.jp2')
        if files:
            return files[0]
        
    raise FileNotFoundError(f'Failed to detect {band} in zip file.')


def extract_bands_from_zip_file(zipfile: str, output_folder: str, bands: List[str]):
    with ZipFile(file=zipfile) as zip_ref:
        files = zip_ref.namelist()

        # Extract bands
        for band in bands:
            filepath = find_band(file_list=files, band=band)

            # Get zip_info object, change filename
            zip_info = zip_ref.getinfo(name=filepath)
            zip_info.filename = f'{band}.jp2'
            zip_ref.extract(member=zip_info, path=output_folder)

def extract_cloudmsk_from_zip_file(zipfile: str, output_folder: str, counter: int):
    with ZipFile(file=zipfile) as zip_ref:
        files = zip_ref.namelist()

        # Extract cloud image
        pattern = '**CLDPRB_20m**.jp2'
        filepath = fnmatch.filter(names=files, pat=pattern)[0]
        zip_info = zip_ref.getinfo(name=filepath)
        zip_info.filename = f'mask_{str(counter).zfill(2)}.jp2'
        zip_ref.extract(member=zip_info, path=output_folder)


def merge_bands(R: str, G: str, B: str, output_file: str) -> None:
    with tempfile.NamedTemporaryFile(suffix='.vrt') as fp:
        gdal.BuildVRT(
            destName=fp.name,
            srcDSOrSrcDSTab=[R, G, B],
            options=gdal.BuildVRTOptions(
                separate=True,
                resolution='highest'
            )
        )

        gdal.Translate(
            destName=output_file,
            srcDS=fp.name,
            options=gdal.TranslateOptions(
                outputType=gdalconst.GDT_UInt16
            )
        )

def copy_file_change_extension(input_path: str, output_path: str) -> None:
    gdal.Warp(
        destNameOrDestDS=output_path,
        srcDSOrSrcDSTab=input_path,
        options=gdal.WarpOptions(
            warpOptions = "NUM_THREADS=ALL_CPUS",
            dstSRS='EPSG:4126'
        )
    )


def extract(input_folder: str, input_zipfiles: list[str], output_folder: str) -> None:
    """
    Extract, merge bands and reproject rasters.

    These processes have been unified in order to save time, since gdal supports doing
    all the actions at once.
    """
    
    # Output image counter index (example: image_01.tiff)
    counter = 0

    # Bands to extract
    bands = ['B02', 'B03', 'B04', 'B08', 'TCI']

    for zipfile in tqdm(input_zipfiles):
        zipfile = f'{input_folder}/{zipfile}.zip'
        with tempfile.TemporaryDirectory() as tmp_dir:
            extract_bands_from_zip_file(zipfile=zipfile, output_folder=tmp_dir, bands=bands)
            extract_cloudmsk_from_zip_file(zipfile=zipfile, output_folder=tmp_dir, counter=counter)

            # Produce True Color Image
            # Not needed, provided int8 based version works fine
            # merge_bands(
            #     R=f'{tmp_dir}/B04.jp2', 
            #     G=f'{tmp_dir}/B03.jp2', 
            #     B=f'{tmp_dir}/B02.jp2', 
            #     output_file=f'{output_folder}/image_TCI_{str(counter).zfill(2)}.tiff'
            # )

            # Produce False Color Image, change coord system
            merge_bands(
                R=f'{tmp_dir}/B08.jp2',
                G=f'{tmp_dir}/B04.jp2',
                B=f'{tmp_dir}/B03.jp2',
                output_file=f'{tmp_dir}/FCI.jp2'
            )

            # Copy FCI image, change coord system
            copy_file_change_extension(
                input_path=f'{tmp_dir}/FCI.jp2',
                output_path=f'{output_folder}/image_FCI_{str(counter).zfill(2)}.tiff'
            )


            # Copy TCI image, change coord system
            copy_file_change_extension(
                input_path=f'{tmp_dir}/TCI.jp2',
                output_path=f'{output_folder}/image_TCI_{str(counter).zfill(2)}.tiff'
            )

            # Copy mask image, change coord system
            copy_file_change_extension(
                input_path=f'{tmp_dir}/mask_{str(counter).zfill(2)}.jp2',
                output_path=f'{output_folder}/mask_{str(counter).zfill(2)}.tiff'
            )


            
        # Append the counter
        counter += 1


def main1():
    # Iterate folder of Sentinel-2 products, extract them into a single folder

    input_folder = '/home/arminius/repos/sentinel-downloader/rasters'
    input_zipfiles = [
        'S2A_MSIL2A_20220605T093041_N0400_R136_T35VLC_20220605T142109',
        'S2A_MSIL2A_20220605T093041_N0400_R136_T34VFH_20220605T142109',
        'S2A_MSIL2A_20220604T100031_N0400_R122_T34UEG_20220604T141210'
    ]

    output_folder = 'rasters/'

    if os.path.exists(output_folder):
        shutil.rmtree(path=output_folder)
    
    # Create output folder
    os.makedirs(name=output_folder)

    extract(
        input_folder=input_folder,
        input_zipfiles=input_zipfiles,
        output_folder=output_folder
    )



def main():
    input_file = '/home/arminius/repos/sentinel-extract/S2A_MSIL2A_20220605T093041_N0400_R136_T35VLC_20220605T142109.zip'

    bands = ['B02', 'B03', 'B04', 'B06', 'B08']
    output_folder = '/tmp/test'

    extract_bands_from_zip_file(zipfile=input_file, output_folder=output_folder, bands=bands)
    extract_cloudmsk_from_zip_file(zipfile=input_file, output_folder=output_folder, counter=0)

    # Produce True Color Image
    merge_bands(
        R=f'{output_folder}/B04.jp2', 
        G=f'{output_folder}/B03.jp2', 
        B=f'{output_folder}/B02.jp2', 
        output_file=f'{output_folder}/TCI.tiff'
    )

    # Produce False Color Image
    merge_bands(
        R=f'{output_folder}/B08.jp2', 
        G=f'{output_folder}/B04.jp2', 
        B=f'{output_folder}/B03.jp2', 
        output_file=f'{output_folder}/FCI.tiff'
    )

    # Produce Weird Color Image
    merge_bands(
        R=f'{output_folder}/B08.jp2',
        G=f'{output_folder}/B06.jp2',
        B=f'{output_folder}/B04.jp2',
        output_file=f'{output_folder}/WCI.tiff'
    )


if __name__ == '__main__':
    main1()
