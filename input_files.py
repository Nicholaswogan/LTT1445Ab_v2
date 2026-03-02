import warnings
import os
import shutil
warnings.filterwarnings('ignore')
THISFILE = os.path.dirname(os.path.abspath(__file__))
os.environ["picaso_refdata"] = os.path.join(THISFILE, "picasofiles", "reference")
os.environ["PYSYN_CDBS"] = os.path.join(THISFILE, "picasofiles", "reference", "stellar_grids")

import urllib.parse
import urllib.request
import zipfile

def download_and_unzip(zip_url, destination_folder, zip_filename=None, delete_zip=False, show_progress=True):
    os.makedirs(destination_folder, exist_ok=True)

    if zip_filename is None:
        zip_filename = os.path.basename(urllib.parse.urlparse(zip_url).path) or "download.zip"

    zip_path = os.path.join(destination_folder, zip_filename)

    def _progress_hook(block_num, block_size, total_size):
        if not show_progress:
            return
        downloaded = block_num * block_size
        if total_size > 0:
            downloaded = min(downloaded, total_size)
            percent = (downloaded / total_size) * 100.0
            print(f"\rDownloading {zip_filename}: {percent:6.2f}% ({downloaded}/{total_size} bytes)", end="", flush=True)
        else:
            print(f"\rDownloading {zip_filename}: {downloaded} bytes", end="", flush=True)

    urllib.request.urlretrieve(zip_url, zip_path, reporthook=_progress_hook)
    if show_progress:
        print()

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination_folder)

    if delete_zip:
        os.remove(zip_path)

    return zip_path

def main():
    reference_dir = 'picasofiles/reference'
    reference_installed = os.path.isdir(reference_dir) and bool(os.listdir(reference_dir))
    if not reference_installed:
        download_and_unzip(
            zip_url='https://github.com/natashabatalha/picaso/archive/refs/tags/v4.0.zip',
            destination_folder='tmp',
            delete_zip=True
        )
        shutil.move(os.path.join('tmp', 'picaso-4.0', 'reference'), 'picasofiles')
        shutil.rmtree('tmp', ignore_errors=True)
    else:
        print('Picaso reference is already downloaded')

    url = "https://zenodo.org/records/17381172/files/opacities_photochem_0.1_250.0_R15000.db.zip"
    if not os.path.exists('picasofiles/opacities_photochem_0.1_250.0_R15000.db'):
        download_and_unzip(
            zip_url=url, 
            destination_folder='picasofiles', 
            delete_zip=True,
        )
    else:
        print('Opacity files is already downloaded')

    from photochem.extensions import hotrocks
    hotrocks.download_sphinx_spectra(
        filename='inputs/sphinx.h5'
    )

if __name__ == '__main__':
    main()
