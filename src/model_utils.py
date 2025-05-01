# model_utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from config import SEQ_LEN, EMBED_DIM, LSTM_UNITS, DROPOUT, NUM_CHORDS
from tqdm import tqdm

def build_melody_model(num_genres, num_moods, lower_bound=None, upper_bound=None):
    span = (upper_bound or 102) - (lower_bound or 24)

    note_input  = layers.Input(shape=(SEQ_LEN,), dtype='int32', name="note_seq")
    genre_input = layers.Input(shape=(),       dtype='int32', name="genre_id")
    mood_input  = layers.Input(shape=(),       dtype='int32', name="mood_id")

    note_emb = layers.Embedding(input_dim=span, output_dim=EMBED_DIM)(note_input)
    g_emb    = layers.Embedding(input_dim=num_genres, output_dim=EMBED_DIM)(genre_input)
    m_emb    = layers.Embedding(input_dim=num_moods,  output_dim=EMBED_DIM)(mood_input)

    g_seq = layers.RepeatVector(SEQ_LEN)(g_emb)
    m_seq = layers.RepeatVector(SEQ_LEN)(m_emb)

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

        preds = np.clip(preds, 1e-8, 1-1e-8)
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

    cg_seq = layers.Lambda(lambda inputs: tf.tile(tf.expand_dims(inputs[0],1), [1, tf.shape(inputs[1])[1], 1]))([cg_emb, chord_input])
    cm_seq = layers.Lambda(lambda inputs: tf.tile(tf.expand_dims(inputs[0],1), [1, tf.shape(inputs[1])[1], 1]))([cm_emb, chord_input])

    y = layers.Concatenate(axis=-1)([chord_input, cg_seq, cm_seq])
    y = layers.LSTM(LSTM_UNITS, return_sequences=True)(y)
    y = layers.Dropout(DROPOUT)(y)
    y = layers.LSTM(LSTM_UNITS, return_sequences=True)(y)
    y = layers.Dropout(DROPOUT)(y)

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
