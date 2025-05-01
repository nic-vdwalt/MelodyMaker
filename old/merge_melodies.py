import numpy as np
from tqdm import tqdm
import glob
import msgpack
import midi_tools

#files = glob.glob('{}/*.mid*'.format(path))
try:
    files = glob.glob('generated*.mid*')
except Exception as e:
    raise e

songs = np.zeros((0,156))
for f in tqdm(files):
    try:
        song = np.array(midi_tools.midiToNoteStateMatrix(f))

        if np.array(song).shape[0] > 10:
            songs.append(song)
            songs = np.concatenate((songs,song))
    except Exception as e:
        raise e

midi_tools.noteStateMatrixToMidi(songs, "final")