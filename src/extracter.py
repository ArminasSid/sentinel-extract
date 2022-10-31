import argparse
import fnmatch
import os, glob, shutil
import zipfile
import pandas as pd
import tempfile
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='Extraction of sentinel-2 products')
    # model and dataset
    parser.add_argument("-i", "--input_products", type=str, default='products.csv',
                        help='Path to products file, contains product descriptions (default: products.csv)')
    parser.add_argument("-f", "--input_folder", type=str, default='products',
                        help='Folder of products (default: products)')
    parser.add_argument("-l", "--longtitude", type=str, default='long.txt',
                        help='Longtitude file, must use T before every point, |T34,T35| (default: long.txt)')
    parser.add_argument("-p", "--pull_mask", type=int, default=0,
                        help="Extract type of mask, 0 for jp2, 1 for gml. (default: 0)")
    parser.add_argument("-t", "--type", type=str, default="TCI",
                        help="Type of image to extract. (default: TCI)")
    parser.add_argument("-o", "--output_folder", type=str, default="output",
                        help="Output folder. (default: output)")

    args = parser.parse_args()

    if not os.path.exists(args.input_products):
        raise FileNotFoundError("--input_products file not found.")
    if not os.path.exists(args.input_folder):
        raise NotADirectoryError("--input_folder not found.")
    if not os.path.exists(args.longtitude):
        raise FileNotFoundError("--longtitude file not found.")
    if args.pull_mask != 1 and args.pull_mask != 0:
        raise IndexError("--pull_mask index out of range.")

    return args

number = 0
# path = 'C:/pythonStuff/sentinel/products2020'


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print('Folder {} has been created.'.format(folder_path))

# for filename in os.listdir(path):
def pull_products1(path, image_type, filename, folders):
    global number
    for folder in folders:
        if fnmatch.fnmatch(filename, '*{}*'.format(folder)):
            src = "{}/{}".format(path, filename)
            dst = folder
            with tempfile.TemporaryDirectory() as tmpdirname:
                print(tmpdirname)
                temp_location = tmpdirname
                with zipfile.ZipFile(src, 'r') as zip_ref:
                    zip_ref.extractall(temp_location)
                filelist = glob.glob(os.path.join(temp_location, "*"))
                path_to_img = ''
                for f in filelist:
                    path_to_img = "{}/GRANULE".format(f)
                    path_to_msk = path_to_img       
                filelist = glob.glob(os.path.join(path_to_img, "*"))
                for f in filelist:
                    path_to_img = "{}/IMG_DATA/R10m".format(f)
                    path_to_msk = "{}/QI_DATA".format(f)
                image_name = ''
                for file in os.listdir(path_to_img):
                    if fnmatch.fnmatch(file, '*_{}_*'.format(image_type)):
                        image_name = file
                        path_to_img = "{}/{}".format(path_to_img, file)
                for file in os.listdir(path_to_msk):
                    if fnmatch.fnmatch(file, 'MSK_CLOUDS_B00.gml'):
                        path_to_msk = "{}/{}".format(path_to_msk, file)
                shutil.copy(path_to_img, dst)
                shutil.copy(path_to_msk, dst)
                os.rename('{}/{}'.format(dst, image_name), "{}/image_{}.jp2".format(dst, number))
                os.rename('{}/{}'.format(dst, 'MSK_CLOUDS_B00.gml'), "{}/mask_{}.gml".format(dst, number))
                number += 1
#                 print('Image: {} and mask: {} created.'.format("image_{}.jp2".format(number), "mask_{}.gml".format(number)))
                return
            

# for filename in os.listdir(path):
def pull_products0(path, image_type, filename, folders):
    global number
    for folder in folders:
        if fnmatch.fnmatch(filename, '*{}*'.format(folder[folder.rfind('/')+1:])):
            src = "{}/{}".format(path, filename)
            dst = folder
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_location = tmpdirname
                with zipfile.ZipFile(src, 'r') as zip_ref:
                    zip_ref.extractall(temp_location)
                filelist = glob.glob(os.path.join(temp_location, "*"))
                path_to_img = ''
                for f in filelist:
                    path_to_img = "{}/GRANULE".format(f)
                    path_to_msk = path_to_img       
                filelist = glob.glob(os.path.join(path_to_img, "*"))
                for f in filelist:
                    path_to_img = "{}/IMG_DATA/R10m".format(f)
                    path_to_msk = "{}/QI_DATA".format(f)
                image_name = ''
                for file in os.listdir(path_to_img):
                    if fnmatch.fnmatch(file, '*_{}_*'.format(image_type)):
                        image_name = file
                        path_to_img = "{}/{}".format(path_to_img, file)
                for file in os.listdir(path_to_msk):
                    if fnmatch.fnmatch(file, 'MSK_CLDPRB_20m.jp2'):
                        path_to_msk = "{}/{}".format(path_to_msk, file)
                shutil.copy(path_to_img, dst)
                shutil.copy(path_to_msk, dst)
                os.rename('{}/{}'.format(dst, image_name), "{}/image_{}.jp2".format(dst, number))
                os.rename('{}/{}'.format(dst, 'MSK_CLDPRB_20m.jp2'), "{}/mask_{}.jp2".format(dst, number))
                number += 1
#                 print('Image: {} and mask: {} created.'.format("image_{}.jp2".format(number), "mask_{}.jp2".format(number)))
                return
 


if __name__=='__main__':
    args = parse_arguments()

    folders = []
    with open(args.longtitude) as f:
        line = f.readline()
        for element in line.split(','):
            folders.append("{}/{}".format(args.output_folder, element))

    for folder in folders:
        check_folder(folder)

    products_df = pd.read_csv(args.input_products, index_col=0)
    products_df_sorted = products_df.sort_values(['cloudcoverpercentage', 'ingestiondate'], ascending=[True, True])
    for product in tqdm(products_df_sorted['title'].tolist()):
        if args.pull_mask == 0:
            pull_products0(args.input_folder, args.type, '{}.zip'.format(product), folders)
        elif args.pull_mask == 1:
            pull_products1(args.input_folder, args.type, '{}.zip'.format(product), folders)
