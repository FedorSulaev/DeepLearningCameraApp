import os
import sys
import tarfile
from six.moves.urllib.request import urlretrieve

data_folder = 'data/'

def download_data(filename, url, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    file_path = data_folder + filename
    if force or not os.path.exists(file_path):
        filename, _ = urlretrieve(url + filename, file_path)
    statinfo = os.stat(file_path)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
          'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return file_path

def extract_data(filename, force=False):
    # remove .tar.gz
    root = data_folder + os.path.splitext(os.path.splitext(filename)[0])[0]
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(data_folder + filename)
        sys.stdout.flush()
        tar.extractall(data_folder)
        tar.close()
    return root