import os
import glob
import argparse
import pickle

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import midi_tools       # your helper module
import filter as gm     # provides Genres() and Moods()
import report as r      # for reporting

# 1) PARAMETERS
SEQ_LEN              = 60
EMBED_DIM            = 16
LSTM_UNITS           = 128
DROPOUT              = 0.3
BATCH_SIZE           = 64
EPOCHS               = 20
MIDI_DIR             = 'Rock_Music_Midi'
SEED_PATH            = 'seed.pkl'
MELODY_MODEL_PATH    = 'melody_model.h5'
CHORD_MODEL_PATH     = 'chord_model.h5'


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
    Load MIDI files, build classification training data (pitch indices).
    Returns:
    - X_notes: np.ndarray (n_samples, seq_len)
    - y_notes: np.ndarray (n_samples,)
    - X_genre: np.ndarray (n_samples,)
    - X_mood:  np.ndarray (n_samples,)
    """
    files = glob.glob(os.path.join(midi_dir, '*.mid*'))
    if not files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}")

    span = upper_bound - lower_bound
    raw_X, raw_y, X_genre, X_mood = [], [], [], []

    for f in files:
        mat = np.array(midi_tools.midi_to_note_state_matrix(f, lower_bound, upper_bound))
        pitches = np.argmax(mat[..., 0], axis=1)
        if len(pitches) <= seq_len:
            continue

        for i in range(len(pitches) - seq_len):
            raw_X.append(pitches[i : i + seq_len])
            raw_y.append(pitches[i + seq_len])
            X_genre.append(genre_enc.transform([genres[0]])[0])
            X_mood.append(mood_enc.transform([moods[0]])[0])

    X_notes = np.array(raw_X, dtype='int32')              # (n, seq_len)
    y_notes = np.array(raw_y, dtype='int32')              # (n,)
    X_genre = np.array(X_genre, dtype='int32')            # (n,)
    X_mood  = np.array(X_mood, dtype='int32')             # (n,)

    return X_notes, y_notes, X_genre, X_mood


def build_melody_model(num_genres, num_moods, lower_bound=24, upper_bound=102):
    span = upper_bound - lower_bound

    note_input  = layers.Input(shape=(SEQ_LEN,), dtype='int32', name="note_seq")
    genre_input = layers.Input(shape=(),       dtype='int32', name="genre_id")
    mood_input  = layers.Input(shape=(),       dtype='int32', name="mood_id")

    note_emb = layers.Embedding(input_dim=span, output_dim=EMBED_DIM)(note_input)    # -> (batch, SEQ_LEN, EMBED_DIM)
    g_emb    = layers.Embedding(input_dim=num_genres, output_dim=EMBED_DIM)(genre_input)
    m_emb    = layers.Embedding(input_dim=num_moods,  output_dim=EMBED_DIM)(mood_input)

    g_seq = layers.RepeatVector(SEQ_LEN)(g_emb)  # -> (batch, SEQ_LEN, EMBED_DIM)
    m_seq = layers.RepeatVector(SEQ_LEN)(m_emb)  # -> (batch, SEQ_LEN, EMBED_DIM)

    x = layers.Concatenate(axis=-1)([note_emb, g_seq, m_seq])
    x = layers.LSTM(LSTM_UNITS, return_sequences=True)(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.LSTM(LSTM_UNITS)(x)
    x = layers.Dropout(DROPOUT)(x)

    melody_out = layers.Dense(span, activation='softmax', name='melody_out')(x)

    model = Model([note_input, genre_input, mood_input], melody_out, name="melody_model")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model


def generate_melody(model, seed_seq, genre_id, mood_id, length=1500, temperature=1.0):
    seq = list(seed_seq.flatten()) if hasattr(seed_seq, 'flatten') else list(seed_seq)
    result = []

    for _ in tqdm(range(length), desc="Generating melody"):
        note_input = np.array([seq[-SEQ_LEN:]], dtype='int32')
        g_input    = np.array([genre_id], dtype='int32')
        m_input    = np.array([mood_id], dtype='int32')
        preds = model.predict([note_input, g_input, m_input], verbose=0)[0]

        eps = 1e-8
        preds = np.clip(preds, eps, 1 - eps)
        log_preds = np.log(preds) / temperature
        log_preds -= np.max(log_preds)
        exp_preds = np.exp(log_preds)
        probs = exp_preds / np.sum(exp_preds)

        if not np.isfinite(probs).all() or np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)

        next_idx = np.random.choice(len(probs), p=probs)
        result.append(int(next_idx))
        seq.append(int(next_idx))

    return np.array(result, dtype='int32')


def build_chord_model(num_genres, num_moods):
    chord_input = layers.Input(shape=(None, 1), name="melody_seq")
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
    parser.add_argument('--train', action='store_true', help="Run training")
    parser.add_argument('--length', type=int, default=1500,
                        help="Length of melody to generate (timesteps)")
    args = parser.parse_args()

    genres, moods, genre_enc, mood_enc = load_label_encoders()
    r.updateData(genres, moods)

    if args.train:
        X_notes, y_notes, X_genre, X_mood = prepare_training_data(
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

        # save seed for generation
        with open(SEED_PATH, 'wb') as f:
            pickle.dump(X_notes[0], f)

        print("✔ Training complete — models and seed saved.")

    else:
        if not (os.path.exists(MELODY_MODEL_PATH) and os.path.exists(CHORD_MODEL_PATH)):
            print("❌ Models not found. Run with --train first:\n    python src/rnn.py --train")
            return

        melody_model = tf.keras.models.load_model(MELODY_MODEL_PATH)
        chord_model  = tf.keras.models.load_model(CHORD_MODEL_PATH)
        with open(SEED_PATH, 'rb') as f:
            seed = pickle.load(f)

        g_id = genre_enc.transform([genres[0]])[0]
        m_id = mood_enc.transform([moods[0]])[0]

        melody = generate_melody(melody_model, seed, g_id, m_id, length=args.length)
        chords = generate_chords(chord_model, melody.reshape(-1, 1), g_id, m_id)

        midi_tools.note_state_matrix_to_midi(
            melody, chords, "output_generated.mid"
        )
        print("✔ Generation complete — MIDI saved as output_generated.mid")


if __name__ == "__main__":
    main()
