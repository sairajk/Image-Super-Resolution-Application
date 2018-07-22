'''
A python program to download files from Google drive given the file's sharable link
#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039
'''
from __future__ import print_function, division, absolute_import
import requests
import zipfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

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
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


# ================================================================================

if __name__ == "__main__":
    # The filename (along with path) which with you want to save the downloaded file
    file_name = 'bbc_part1_LR.zip'

    # file id from the sharable google drive link
    file_id = '11ymxoModCWpaoJcwOvXJwMmYmFHo4Ggf'

    print("Download started")

    download_file_from_google_drive(file_id, file_name)

    print("File downloaded.")

    '''
    # Destination for zip file output
    zip_op_path = "/output"
    
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(zip_op_path)
    zip_ref.close()
    '''
# ================================================================================

'''
# sample sharable link from google drive
https://drive.google.com/open?id=11ymxoModCWpaoJcwOvXJwMmYmFHo4Ggf
id = 11ymxoModCWpaoJcwOvXJwMmYmFHo4Ggf
'''