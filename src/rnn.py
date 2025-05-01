# Improved RNN for Melody + Chord Progression Generation
import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import midi_tools  # your existing helper
import GetGM as gm
import Report as r

# 1) PARAMETERS
SEQ_LEN = 60
FEATURE_DIM = 1             # mono-feature (pitch) for now
EMBED_DIM = 16
LSTM_UNITS = 128
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 20

# 2) LOAD GENRE/MOOD ENCODERS
genres = gm.Genres()
moods  = gm.Moods()
genre_enc = LabelEncoder().fit(genres)
mood_enc  = LabelEncoder().fit(moods)

r.updateData(genres, moods)  # keep your reporting in sync

# 3) DATA LOADER
def load_midi_sequences(path):
    files = glob.glob(os.path.join(path, '*.mid*'))
    seqs = []
    for f in tqdm(files, desc="Loading MIDIs"):
        mat = np.array(midi_tools.midiToNoteStateMatrix(f))
        if mat.shape[0] > SEQ_LEN:
            seqs.append(mat[:,0:1])  # only pitch column
    return seqs

raw_seqs = load_midi_sequences('Rock_Music_Midi')

# 4) BUILD X, y, plus genre/mood features
X_notes, y_notes, X_genre, X_mood = [], [], [], []
for seq in raw_seqs:
    # flatten rolling windows
    for i in range(len(seq) - SEQ_LEN - 1):
        window = seq[i:i+SEQ_LEN]
        target = seq[i+SEQ_LEN]
        X_notes.append(window)
        y_notes.append(target)
        # assign same genre/mood for entire sequence
        X_genre.append( genre_enc.transform([genres[0]])[0] )
        X_mood.append(  mood_enc.transform([moods[0]])[0] )

X_notes = np.array(X_notes).astype('float32')  # (N, SEQ_LEN, 1)
y_notes = np.array(y_notes).astype('float32')  # (N, 1)
X_genre = np.array(X_genre)
X_mood  = np.array(X_mood)

# 5) SCALE
scaler = MinMaxScaler(feature_range=(0,1))
# collapse time axis to scale
flat_X = X_notes.reshape(-1, 1)
scaler.fit(flat_X)
X_notes = scaler.transform(flat_X).reshape(X_notes.shape)
y_notes = scaler.transform(y_notes)

# 6) MELODY MODEL (Functional API to accept genre & mood)
note_input  = layers.Input(shape=(SEQ_LEN, FEATURE_DIM), name="note_seq")
genre_input = layers.Input(shape=(), dtype='int32', name="genre_id")
mood_input  = layers.Input(shape=(), dtype='int32', name="mood_id")

g = layers.Embedding(input_dim=len(genres), output_dim=EMBED_DIM)(genre_input)
m = layers.Embedding(input_dim=len(moods),  output_dim=EMBED_DIM)(mood_input)
# expand and tile to time dimension
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

# 7) TRAIN MELODY MODEL
melody_model.fit(
    [X_notes, X_genre, X_mood],
    y_notes,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    shuffle=True
)

# 8) CHORD MODEL: takes generated melody + genre/mood to predict chord labels
#    assume chord_data preprocessed similarly into one-hot vectors
chord_input = layers.Input(shape=(SEQ_LEN, FEATURE_DIM), name="melody_seq")
cg = layers.Input(shape=(), dtype='int32', name="ch_genre_id")
cm = layers.Input(shape=(), dtype='int32', name="ch_mood_id")

cg_emb = layers.Embedding(len(genres), EMBED_DIM)(cg)
cm_emb = layers.Embedding(len(moods), EMBED_DIM)(cm)
cg_seq = layers.RepeatVector(SEQ_LEN)(cg_emb)
cm_seq = layers.RepeatVector(SEQ_LEN)(cm_emb)

y = layers.Concatenate(axis=-1)([chord_input, cg_seq, cm_seq])
y = layers.LSTM(LSTM_UNITS, return_sequences=True)(y)
y = layers.Dropout(DROPOUT)(y)
y = layers.LSTM(LSTM_UNITS)(y)
y = layers.Dropout(DROPOUT)(y)
# suppose NUM_CHORDS classes
NUM_CHORDS = 12  
chord_out = layers.Dense(NUM_CHORDS, activation='softmax')(y)

chord_model = Model([chord_input, cg, cm], chord_out, name="chord_model")
chord_model.compile(optimizer='adam', loss='categorical_crossentropy')

# ... similarly train chord_model on your chord dataset ...

# 9) GENERATION FUNCTIONS
def generate_melody(seed_seq, genre_id, mood_id, length=100):
    seq = seed_seq.copy()
    out = []
    for _ in range(length):
        p = melody_model.predict([
            seq[np.newaxis,...], np.array([genre_id]), np.array([mood_id])
        ])[0]
        out.append(p)
        seq = np.vstack([seq[1:], p[np.newaxis,...]])
    return np.array(out)

def generate_chords(melody_seq, genre_id, mood_id):
    preds = chord_model.predict([
        melody_seq[np.newaxis,...],
        np.array([genre_id]),
        np.array([mood_id])
    ])[0]
    # pick argmax chord for each step
    return np.argmax(preds, axis=-1)

# 10) RUN & SAVE MIDI
seed = X_notes[0]  # your starting window
g_id, m_id = genre_enc.transform([genres[0]])[0], mood_enc.transform([moods[0]])[0]
melody = generate_melody(seed, g_id, m_id, length=200)
chords = generate_chords(melody, g_id, m_id)

# inverse‐scale melody
melody = scaler.inverse_transform(melody)

midi_tools.sequence_to_midi(melody, chords, "output_generated.mid")
