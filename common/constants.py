# Sebastian Thomas (datascience at sebastianthomas dot de)

# paths
from pathlib import Path


COMMANDS = ('yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop',
            'go')
UNKNOWN_CATEGORY = 'unknown'
SILENCE_CATEGORY = 'silence'
CATEGORIES = COMMANDS + (UNKNOWN_CATEGORY, SILENCE_CATEGORY)
NUM_CATEGORIES = len(CATEGORIES) - 1
# disregard SILENCE_CATEGORY for now (future work)

NUM_SAMPLES = 16000

DATA_DIR = Path('data', 'speech_commands', '0.0.2')
CLASSIFIER_PATH = 'classifier.tf'
