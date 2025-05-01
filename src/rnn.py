# rnn.py

# Improved RNN for Melody + Chord Progression Generation

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
import midi_tools       # your existing helper module
import filter as gm     # provides Genres() and Moods()
import report as r      # for reporting

# 1) PARAMETERS
SEQ_LEN     = 60
FEATURE_DIM = 1             # mono-feature (pitch) for now
EMBED_DIM   = 16
LSTM_UNITS  = 128
DROPOUT     = 0.3
BATCH_SIZE  = 64
EPOCHS      = 20
MIDI_DIR    = 'Rock_Music_Midi'

# 2) LOAD GENRE/MOOD ENCODERS
def load_label_encoders():
    raw_genres = gm.Genres()
    raw_moods  = gm.Moods()

    # Ensure genres and moods are iterable lists
    if isinstance(raw_genres, str) or not hasattr(raw_genres, '__iter__'):
        raw_genres = [raw_genres]
    if isinstance(raw_moods, str) or not hasattr(raw_moods, '__iter__'):
        raw_moods = [raw_moods]

    if not raw_genres or not raw_moods:
        raise ValueError("gm.Genres() or gm.Moods() returned empty.")

    genre_enc = LabelEncoder().fit(raw_genres)
    mood_enc  = LabelEncoder().fit(raw_moods)

    return raw_genres, raw_moods, genre_enc, mood_enc

# 3) PREPARE TRAINING DATA
def prepare_training_data(
    midi_dir: str,
    genres: list,
    moods: list,
    genre_enc: LabelEncoder,
    mood_enc: LabelEncoder,
    seq_len: int = SEQ_LEN,
    feature_dim: int = FEATURE_DIM
):
    files = glob.glob(os.path.join(midi_dir, '*.mid*'))
    if not files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    seqs = []
    for f in files:
        mat = np.array(midi_tools.midi_to_note_state_matrix(f))
        if mat.ndim == 3:
            mat = mat[..., 0]
        if mat.shape[0] > seq_len:
            seqs.append(mat[:, :feature_dim])

    if not seqs:
        raise ValueError(f"No MIDI sequences longer than SEQ_LEN={seq_len} in {midi_dir}")

    X_notes, y_notes, X_genre, X_mood = [], [], [], []
    for seq in seqs:
        for i in range(len(seq) - seq_len):
            window = seq[i : i + seq_len]
            target = seq[i + seq_len, 0].item()

            X_notes.append(window)
            y_notes.append(target)
            X_genre.append(genre_enc.transform([genres[0]])[0])
            X_mood.append(mood_enc.transform([moods[0]])[0])

    X_notes = np.array(X_notes, dtype='float32')
    y_notes = np.array(y_notes, dtype='float32').reshape(-1, 1)
    X_genre = np.array(X_genre, dtype='int32')
    X_mood  = np.array(X_mood, dtype='int32')

    flat_X = X_notes.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(flat_X)
    X_notes = scaler.transform(flat_X).reshape(X_notes.shape)
    y_notes = scaler.transform(y_notes)

    return X_notes, y_notes, X_genre, X_mood, scaler

# 4) LOAD ENCODERS & DATA
genres, moods, genre_enc, mood_enc = load_label_encoders()
r.updateData(genres, moods)
X_notes, y_notes, X_genre, X_mood, scaler = prepare_training_data(
    MIDI_DIR, genres, moods, genre_enc, mood_enc
)

# 5) MELODY MODEL
note_input  = layers.Input(shape=(SEQ_LEN, FEATURE_DIM), name="note_seq")
genre_input = layers.Input(shape=(), dtype='int32', name="genre_id")
mood_input  = layers.Input(shape=(), dtype='int32', name="mood_id")

g = layers.Embedding(input_dim=len(genres), output_dim=EMBED_DIM)(genre_input)
m = layers.Embedding(input_dim=len(moods),  output_dim=EMBED_DIM)(mood_input)
g_seq = layers.RepeatVector(SEQ_LEN)(g)
m_seq = layers.RepeatVector(SEQ_LEN)(m)

x = layers.Concatenate(axis=-1)([note_input, g_seq, m_seq])
x = layers.LSTM(LSTM_UNITS, return_sequences=True)(x)
x = layers.Dropout(DROPOUT)(x)
x = layers.LSTM(LSTM_UNITS)(x)
x = layers.Dropout(DROPOUT)(x)
melody_out = layers.Dense(FEATURE_DIM, activation='linear')(x)

melody_model = Model([note_input, genre_input, mood_input], melody_out, name="melody_model")
melody_model.compile(optimizer='adam', loss='mse')

# 6) TRAIN MELODY MODEL
melody_model.fit(
    [X_notes, X_genre, X_mood],
    y_notes,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    shuffle=True
)

# 7) CHORD MODEL (now supports variable-length inputs and outputs per timestep)
chord_input = layers.Input(shape=(None, FEATURE_DIM), name="melody_seq")
cg = layers.Input(shape=(), dtype='int32', name="ch_genre_id")
cm = layers.Input(shape=(), dtype='int32', name="ch_mood_id")

cg_emb = layers.Embedding(input_dim=len(genres), output_dim=EMBED_DIM)(cg)
cm_emb = layers.Embedding(input_dim=len(moods),  output_dim=EMBED_DIM)(cm)

# Tile embeddings to match the melody sequence length dynamically
cg_seq = layers.Lambda(
    lambda inputs: tf.tile(tf.expand_dims(inputs[0], 1), [1, tf.shape(inputs[1])[1], 1])
)([cg_emb, chord_input])
cm_seq = layers.Lambda(
    lambda inputs: tf.tile(tf.expand_dims(inputs[0], 1), [1, tf.shape(inputs[1])[1], 1])
)([cm_emb, chord_input])

y = layers.Concatenate(axis=-1)([chord_input, cg_seq, cm_seq])
y = layers.LSTM(LSTM_UNITS, return_sequences=True)(y)
y = layers.Dropout(DROPOUT)(y)
y = layers.LSTM(LSTM_UNITS, return_sequences=True)(y)
y = layers.Dropout(DROPOUT)(y)

NUM_CHORDS = 12
chord_out = layers.TimeDistributed(
    layers.Dense(NUM_CHORDS, activation='softmax')
)(y)

chord_model = Model([chord_input, cg, cm], chord_out, name="chord_model")
chord_model.compile(optimizer='adam', loss='categorical_crossentropy')

# 8) GENERATION FUNCTIONS
def generate_melody(seed_seq, genre_id, mood_id, length=100):
    seq = seed_seq.copy()
    out = []
    for _ in range(length):
        p = melody_model.predict([
            seq[np.newaxis, ...],
            np.array([genre_id]),
            np.array([mood_id])
        ], verbose=0)[0]
        out.append(p)
        seq = np.vstack([seq[1:], p[np.newaxis, ...]])
    return np.array(out)

def generate_chords(melody_seq, genre_id, mood_id):
    preds = chord_model.predict([
        melody_seq[np.newaxis, ...],
        np.array([genre_id]),
        np.array([mood_id])
    ], verbose=0)[0]
    return np.argmax(preds, axis=-1)

# 9) RUN & SAVE MIDI
if __name__ == "__main__":
    seed = X_notes[0]
    g_id, m_id = genre_enc.transform([genres[0]])[0], mood_enc.transform([moods[0]])[0]
    melody = generate_melody(seed, g_id, m_id, length=200)
    chords = generate_chords(melody, g_id, m_id)
    melody = scaler.inverse_transform(melody)
    midi_tools.note_state_matrix_to_midi(melody, chords, "output_generated.mid")

