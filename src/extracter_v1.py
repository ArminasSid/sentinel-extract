from osgeo import gdal, gdalconst
from zipfile import ZipFile
import fnmatch
from typing import List
import tempfile
import pandas as pd
from tqdm import tqdm


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
        zip_info.filename = f'mask_{str(counter):2}.jp2'
        zip_ref.extract(member=zip_info, path=output_folder)


def merge_bands(R: str, G: str, B: str, output_file: str) -> None:
    with tempfile.NamedTemporaryFile(suffix='.vrt') as fp:
        gdal.BuildVRT(
            destName=fp.name,
            srcDSOrSrcDSTab=[R, G, B],
            options=gdal.BuildVRTOptions(
                separate=True,
                outputSRS='EPSG:4126'
            )
        )

        gdal.Translate(
            destName=output_file,
            srcDS=fp.name,
            options=gdal.TranslateOptions(
                outputType=gdalconst.GDT_UInt16
            )
        )


def extract(input_folder: str, input_zipfiles: str, output_folder: str) -> None:
    # Output image counter index (example: image_01.tiff)
    counter = 0

    # Bands to extract
    bands = ['B02', 'B03', 'B04', 'B08']


    for zipfile in tqdm(input_zipfiles):
        zipfile = f'{input_folder}/{zipfile}.zip'
        with tempfile.gettempdir() as tmp_dir:
            extract_bands_from_zip_file(zipfile=zipfile, output_folder=tmp_dir, bands=bands)
            extract_cloudmsk_from_zip_file(zipfile=zipfile, output_folder=output_folder, counter=counter)

            # Produce True Color Image
            merge_bands(
                R=f'{tmp_dir}/B04.jp2', 
                G=f'{tmp_dir}/B03.jp2', 
                B=f'{tmp_dir}/B02.jp2', 
                output_file=f'{output_folder}/image_TCI_{str(counter):2}.tiff'
            )

            # Produce False Color Image
            merge_bands(
                R=f'{tmp_dir}/B08.jp2', 
                G=f'{tmp_dir}/B04.jp2', 
                B=f'{tmp_dir}/B03.jp2', 
                output_file=f'{output_folder}/image_FCI_{str(counter):2}.tiff'
            )




def main():
    input_file = '/home/arminius/repos/sentinel-extract/S2A_MSIL2A_20220605T093041_N0400_R136_T35VLC_20220605T142109.zip'

    bands = ['B02', 'B03', 'B04', 'B08']
    output_folder = '/tmp/test'

    extract_bands_from_zip_file(zipfile=input_file, output_folder=output_folder, bands=bands)
    extract_cloudmsk_from_zip_file(zipfile=input_file, output_folder=output_folder)

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


def main1():
    products_df = pd.read_csv(input_products, index_col=0)
    products_df_sorted = products_df.sort_values(['cloudcoverpercentage', 'ingestiondate'], ascending=[True, True])
    for product in tqdm(products_df_sorted['title'].tolist()):



if __name__ == '__main__':
    main()
