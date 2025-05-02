# data_utils.py
import os, glob
import numpy as np
from sklearn.preprocessing import LabelEncoder
import utils.midi_utils as midi_utils
# import filter as gm
from config import MIDI_DIR, LOWER_BOUND, UPPER_BOUND, SEQ_LEN

def prepare_training_data(
    midi_dir=MIDI_DIR,
    genres=None,
    moods=None,
    genre_enc=None,
    mood_enc=None,
    seq_len=SEQ_LEN,
    lower_bound=LOWER_BOUND,
    upper_bound=UPPER_BOUND
):
    files = glob.glob(os.path.join(midi_dir, '*.mid*'))
    if not files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    span = upper_bound - lower_bound
    raw_X, raw_y, X_genre, X_mood = [], [], [], []

    for f in files:
        mat = np.array(midi_utils.midi_to_note_state_matrix(f, lower_bound, upper_bound))
        pitches = np.argmax(mat[..., 0], axis=1)
        if len(pitches) <= seq_len:
            continue

        for i in range(len(pitches) - seq_len):
            raw_X.append(pitches[i : i + seq_len])
            raw_y.append(pitches[i + seq_len])
            X_genre.append(genre_enc.transform([genres[0]])[0])
            X_mood.append(mood_enc.transform([moods[0]])[0])

    X_notes = np.array(raw_X, dtype='int32')
    y_notes = np.array(raw_y, dtype='int32')
    X_genre = np.array(X_genre, dtype='int32')
    X_mood  = np.array(X_mood, dtype='int32')

    return X_notes, y_notes, X_genre, X_mood
