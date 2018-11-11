import argparse

import logging
import os
import requests
import zipfile

drive_id_dict = {
    'casia-webface': '1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz'
}


def download_and_extract_file(casia_name, data_dir):
    file_id = drive_id_dict[casia_name]
    destination = os.path.join(data_dir, casia_name + '.zip')
    if not os.path.exists(destination):
        print('Downloading file to %s' % destination)
        download_file_from_google_drive(file_id, destination)
        with zipfile.ZipFile(destination, 'r') as zip_ref:
            print('Extracting file to %s' % data_dir)
            zip_ref.extractall(data_dir)


def download_file_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--data-dir', type=str, action='store', default='datasets', dest='data_dir',
                        help='Path to the datasets directory')
    parser.add_argument('--casia-name', type=str, action='store', default='casia-webface', dest='casia_name',
                        help='Name of the CASIA dataset to load')
    args = parser.parse_args()
    download_and_extract_file(args.casia_name, args.data_dir)
