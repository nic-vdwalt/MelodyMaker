# utils/model_utils.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from config import SEQ_LEN, EMBED_DIM, LSTM_UNITS, DROPOUT, NUM_CHORDS
from utils.key_utils import key_enc

def build_melody_model(lower_bound=None, upper_bound=None):
    span = (upper_bound or 102) - (lower_bound or 24)
    num_keys = len(key_enc.classes_)

    note_input = layers.Input(shape=(SEQ_LEN,), dtype='int32', name="note_seq")
    key_input  = layers.Input(shape=(1,),     dtype='int32', name="key_id")

    note_emb = layers.Embedding(input_dim=span, output_dim=EMBED_DIM)(note_input)
    k_emb    = layers.Embedding(input_dim=num_keys, output_dim=EMBED_DIM)(key_input)
    k_seq    = layers.RepeatVector(SEQ_LEN)(k_emb)

    x = layers.Concatenate(axis=-1)([note_emb, k_seq])
    x = layers.LSTM(LSTM_UNITS, return_sequences=True)(x)
    x = layers.Dropout(DROPOUT)(x)
    x = layers.LSTM(LSTM_UNITS)(x)
    x = layers.Dropout(DROPOUT)(x)

    melody_out = layers.Dense(span, activation='softmax', name='melody_out')(x)
    model = Model([note_input, key_input], melody_out, name="melody_model")
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model

def generate_melody(model, seed_seq, key_id=None, length=1500, temperature=1.0):
    """
    Will automatically detect whether `model` wants 3 inputs
    (note_seq, genre_id, mood_id) or 2 inputs (note_seq, key_id).
    """
    seq = list(seed_seq.flatten()) if hasattr(seed_seq, 'flatten') else list(seed_seq)
    result = []
    n_inputs = len(model.inputs)

    for _ in range(length):
        note_input = np.array([seq[-SEQ_LEN:]], dtype='int32')

        if n_inputs == 3:
            # old model: [note_seq, genre, mood]
            g_input = np.zeros((1,), dtype='int32')
            m_input = np.zeros((1,), dtype='int32')
            preds = model.predict([note_input, g_input, m_input], verbose=0)[0]
        else:
            # new model: [note_seq, key_id]
            k_input = np.array([[key_id]], dtype='int32')
            preds = model.predict([note_input, k_input], verbose=0)[0]

        # temperature sampling
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

def build_chord_model(lower_bound=None, upper_bound=None):
    span = (upper_bound or 102) - (lower_bound or 24)
    num_keys = len(key_enc.classes_)

    chord_input = layers.Input(shape=(None, 1), name="melody_seq")
    key_input   = layers.Input(shape=(1,),     dtype='int32', name="key_id")

    k_emb = layers.Embedding(input_dim=num_keys, output_dim=EMBED_DIM)(key_input)
    k_seq = layers.Lambda(
        lambda inputs: tf.tile(
            tf.expand_dims(inputs[0], 1),
            [1, tf.shape(inputs[1])[1], 1]
        )
    )([k_emb, chord_input])

    y = layers.Concatenate(axis=-1)([chord_input, k_seq])
    y = layers.LSTM(LSTM_UNITS, return_sequences=True)(y)
    y = layers.Dropout(DROPOUT)(y)
    y = layers.LSTM(LSTM_UNITS, return_sequences=True)(y)
    y = layers.Dropout(DROPOUT)(y)

    chord_out = layers.TimeDistributed(
        layers.Dense(NUM_CHORDS, activation='softmax')
    )(y)

    model = Model([chord_input, key_input], chord_out, name="chord_model")
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

def generate_chords(model, melody_seq, key_id=None):
    """
    Detects whether chord_model expects 3 inputs
    (melody_seq, genre_id, mood_id) or 2 inputs (melody_seq, key_id).
    """
    n_inputs = len(model.inputs)

    if n_inputs == 3:
        # old chord model
        cg = np.zeros((1,), dtype='int32')
        cm = np.zeros((1,), dtype='int32')
        preds = model.predict([melody_seq[np.newaxis, ...], cg, cm], verbose=0)[0]
    else:
        preds = model.predict([
            melody_seq[np.newaxis, ...],
            np.array([[key_id]], dtype='int32')
        ], verbose=0)[0]

    return np.argmax(preds, axis=-1)
