import os
import glob
import argparse
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tqdm import tqdm
from numpy.random import choice

import midi_tools       # your helper module
import filter as gm     # provides Genres() and Moods()
import report as r      # for reporting

# 1) PARAMETERS
SEQ_LEN              = 60
FEATURE_DIM          = 1
EMBED_DIM            = 16
LSTM_UNITS           = 128
DROPOUT              = 0.3
BATCH_SIZE           = 64
EPOCHS               = 20
MIDI_DIR             = 'Rock_Music_Midi'
SCALER_PATH          = 'scaler.pkl'
MELODY_MODEL_PATH    = 'melody_model.h5'
CHORD_MODEL_PATH     = 'chord_model.h5'
SEED_PATH            = 'seed.pkl'


def load_label_encoders():
    raw_genres = gm.Genres()
    raw_moods  = gm.Moods()

    if isinstance(raw_genres, str) or not hasattr(raw_genres, '__iter__'):
        raw_genres = [raw_genres]
    if isinstance(raw_moods, str) or not hasattr(raw_moods, '__iter__'):
        raw_moods = [raw_moods]

    if not raw_genres or not raw_moods:
        raise ValueError("gm.Genres() or gm.Moods() returned empty.")

    genre_enc = LabelEncoder().fit(raw_genres)
    mood_enc  = LabelEncoder().fit(raw_moods)
    return raw_genres, raw_moods, genre_enc, mood_enc


def prepare_training_data(
    midi_dir: str,
    genres: list,
    moods: list,
    genre_enc: LabelEncoder,
    mood_enc: LabelEncoder,
    seq_len: int = SEQ_LEN,
    lower_bound: int = 24,
    upper_bound: int = 102
):
    """
    Load MIDI files from `midi_dir`, extract monophonic pitch sequences,
    and build training data for regression (scaled continuous values).

    Returns:
    - X_notes: np.ndarray of shape (n_samples, seq_len, FEATURE_DIM)
    - y_notes: np.ndarray of shape (n_samples, FEATURE_DIM)
    - X_genre: np.ndarray of shape (n_samples,)
    - X_mood: np.ndarray of shape (n_samples,)
    - scaler : fitted MinMaxScaler for inverse transformation
    """
    files = glob.glob(os.path.join(midi_dir, '*.mid*'))
    if not files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    span = upper_bound - lower_bound
    raw_X, raw_y, X_genre, X_mood = [], [], [], []

    for f in files:
        mat = np.array(midi_tools.midi_to_note_state_matrix(f))
        # collapse to monophonic pitch index per timestep
        pitches = np.argmax(mat[..., 0], axis=1)
        if len(pitches) <= seq_len:
            continue

        for i in range(len(pitches) - seq_len):
            window = pitches[i : i + seq_len]
            target = pitches[i + seq_len]

            raw_X.append(window)
            raw_y.append(target)
            X_genre.append(genre_enc.transform([genres[0]])[0])
            X_mood.append(mood_enc.transform([moods[0]])[0])

    # Convert to arrays and reshape for regression model
    X_notes = np.array(raw_X, dtype='float32').reshape(-1, seq_len, FEATURE_DIM)
    y_notes = np.array(raw_y, dtype='float32').reshape(-1, FEATURE_DIM)
    X_genre = np.array(X_genre, dtype='int32')
    X_mood  = np.array(X_mood, dtype='int32')

    # Scale note values to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    flat_inputs = X_notes.flatten().reshape(-1, 1)
    flat_targets = y_notes.flatten().reshape(-1, 1)
    scaler.fit(np.vstack([flat_inputs, flat_targets]))

    X_notes = scaler.transform(flat_inputs).reshape(-1, seq_len, FEATURE_DIM)
    y_notes = scaler.transform(flat_targets).reshape(-1, FEATURE_DIM)

    return X_notes, y_notes, X_genre, X_mood, scaler

def build_melody_model(num_genres, num_moods, lower_bound=24, upper_bound=102):
    span = upper_bound - lower_bound

    note_input  = layers.Input(shape=(SEQ_LEN, FEATURE_DIM), name="note_seq", dtype='int32')
    genre_input = layers.Input(shape=(),         dtype='int32', name="genre_id")
    mood_input  = layers.Input(shape=(),         dtype='int32', name="mood_id")

    # Embed the note sequence and squeeze out the extra FEATURE_DIM axis
    note_emb = layers.Embedding(input_dim=span, output_dim=EMBED_DIM)(note_input)            # -> (batch, SEQ_LEN, 1, EMBED_DIM)
    note_emb = layers.Lambda(lambda x: tf.squeeze(x, axis=2), name="squeeze_note_emb")(note_emb)  # -> (batch, SEQ_LEN, EMBED_DIM)

    # Embed genre and mood IDs and replicate over timesteps
    g_emb = layers.Embedding(input_dim=num_genres, output_dim=EMBED_DIM)(genre_input)  # -> (batch, EMBED_DIM)
    m_emb = layers.Embedding(input_dim=num_moods,  output_dim=EMBED_DIM)(mood_input)   # -> (batch, EMBED_DIM)
    g_seq = layers.RepeatVector(SEQ_LEN)(g_emb)                                      # -> (batch, SEQ_LEN, EMBED_DIM)
    m_seq = layers.RepeatVector(SEQ_LEN)(m_emb)                                      # -> (batch, SEQ_LEN, EMBED_DIM)

    # Concatenate along the feature axis
    x = layers.Concatenate(axis=-1)([note_emb, g_seq, m_seq])  # -> (batch, SEQ_LEN, EMBED_DIM*3)
    x = layers.LSTM(LSTM_UNITS, return_sequences=True)(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.LSTM(LSTM_UNITS)(x)
    x = layers.Dropout(DROPOUT)(x)

    melody_out = layers.Dense(FEATURE_DIM, activation='linear')(x)

    model = Model([note_input, genre_input, mood_input], melody_out, name="melody_model")
    model.compile(optimizer='adam', loss='mse')
    return model

def generate_melody(model, seed_seq, genre_id, mood_id, length=1500, temperature=1.0):
    """
    Generate a new melody sequence using the trained model.

    Args:
        model: Trained Keras melody prediction model.
        seed_seq: Initial sequence of pitches, shape (SEQ_LEN, 1) or (SEQ_LEN,).
        genre_id: Integer genre identifier.
        mood_id: Integer mood identifier.
        length: Number of timesteps to generate.
        temperature: Sampling temperature for diversity.

    Returns:
        numpy.ndarray of generated pitch indices, shape (length,).
    """
    # Flatten seed and convert to int list
    arr = np.array(seed_seq).flatten()
    seq = [int(x) for x in arr]
    result = []

    for _ in tqdm(range(length), desc="Generating melody"):
        # Prepare inputs
        note_input = np.array(seq[-SEQ_LEN:], dtype='int32').reshape(1, SEQ_LEN)
        g_input = np.array([genre_id], dtype='int32')
        m_input = np.array([mood_id], dtype='int32')
        preds = model.predict([note_input, g_input, m_input], verbose=0)[0]

        # Temperature sampling
        eps = 1e-8
        preds = np.clip(preds, eps, 1.0)
        log_preds = np.log(preds) / max(temperature, eps)
        log_preds -= np.max(log_preds)
        exp_preds = np.exp(log_preds)
        probs = exp_preds / np.sum(exp_preds)

        if not np.isfinite(probs).all() or np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)

        next_idx = choice(len(probs), p=probs)
        result.append(int(next_idx))
        seq.append(int(next_idx))

    return np.array(result, dtype='int32')

def build_chord_model(num_genres, num_moods):
    chord_input = layers.Input(shape=(None, FEATURE_DIM), name="melody_seq")
    cg          = layers.Input(shape=(), dtype='int32', name="ch_genre_id")
    cm          = layers.Input(shape=(), dtype='int32', name="ch_mood_id")

    cg_emb = layers.Embedding(input_dim=num_genres, output_dim=EMBED_DIM)(cg)
    cm_emb = layers.Embedding(input_dim=num_moods,  output_dim=EMBED_DIM)(cm)

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
    chord_out = layers.TimeDistributed(layers.Dense(NUM_CHORDS, activation='softmax'))(y)

    model = Model([chord_input, cg, cm], chord_out, name="chord_model")
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def generate_chords(model, melody_seq, genre_id, mood_id):
    preds = model.predict([
        melody_seq[np.newaxis, ...],
        np.array([genre_id]),
        np.array([mood_id])
    ], verbose=0)[0]
    return np.argmax(preds, axis=-1)

def main():
    parser = argparse.ArgumentParser(description="Train or generate melodies and chords")
    parser.add_argument(
        '--train', action='store_true', help="Run training"
    )
    parser.add_argument(
        '--length', type=int, default=1500,
        help="Length of melody to generate (timesteps)"
    )
    args = parser.parse_args()

    genres, moods, genre_enc, mood_enc = load_label_encoders()
    r.updateData(genres, moods)

    if args.train:
        X_notes, y_notes, X_genre, X_mood, scaler = prepare_training_data(
            MIDI_DIR, genres, moods, genre_enc, mood_enc
        )
        melody_model = build_melody_model(len(genres), len(moods))
        chord_model  = build_chord_model(len(genres), len(moods))
        melody_model.fit(
            [X_notes, X_genre, X_mood], y_notes,
            batch_size=BATCH_SIZE, epochs=EPOCHS,
            validation_split=0.1, shuffle=True
        )
        melody_model.save(MELODY_MODEL_PATH)
        chord_model.save(CHORD_MODEL_PATH)
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        with open(SEED_PATH, 'wb') as f:
            pickle.dump(X_notes[0], f)
        print("✔ Training complete — artifacts saved.")

    else:
        if (
            not os.path.exists(MELODY_MODEL_PATH)
            or not os.path.exists(CHORD_MODEL_PATH)
        ):
            print(
                "❌ Models not found. Run with --train first:\n"
                "    python src/rnn.py --train"
            )
            return

        melody_model = tf.keras.models.load_model(MELODY_MODEL_PATH)
        chord_model  = tf.keras.models.load_model(CHORD_MODEL_PATH)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(SEED_PATH, 'rb') as f:
            seed = pickle.load(f)

        g_id = genre_enc.transform([genres[0]])[0]
        m_id = mood_enc.transform([moods[0]])[0]
        melody = generate_melody(
            melody_model, seed, g_id, m_id, length=args.length
        )
        chords = generate_chords(chord_model, melody, g_id, m_id)

        # Reshape for inverse_transform, then flatten back
        melody = scaler.inverse_transform(melody.reshape(-1, 1)).reshape(-1)

        midi_tools.note_state_matrix_to_midi(
            melody, chords, "output_generated.mid"
        )
        print("✔ Generation complete — MIDI saved as output_generated.mid")


if __name__ == "__main__":
    main()

