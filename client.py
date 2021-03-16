# Sebastian Thomas (datascience at sebastianthomas dot de)

# paths
from pathlib import PurePath

# web requests
import requests

# custom modules
from common.constants import API_URL


if __name__ == '__main__':
    for file_path in (PurePath('test', 'bed.wav'),
                      PurePath('test', 'down.wav'),
                      PurePath('test', 'left.wav'),
                      PurePath('test', 'one.wav'),
                      PurePath('test', 'up.wav')):
        with open(file_path, 'rb') as audio_file:
            files = {'audio': (file_path.name, audio_file, 'audio/wav')}
            response = requests.post(API_URL, files=files)

        data = response.json()

        if data.get('message') == 'The keyword is unknown.':
            print('The Keyword Spotting Service did not recognize a valid '
                  'keyword.')
        elif data.get('message') is not None:
            print('The Keyword Spotting Service responded with the following '
                  'message:')
            print(data['message'])
        else:
            print('The Keyword Spotting Service predicted the keyword '
                  '"{}".'.format(data['keyword']))
