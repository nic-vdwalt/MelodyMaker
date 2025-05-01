# generate.py
import argparse
import pickle
import tensorflow as tf
import numpy as np
import midi_tools
from data_utils import load_label_encoders
from model_utils import generate_melody, generate_chords
from config import SEED_PATH, MELODY_MODEL_PATH, CHORD_MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description="Generate melody and chord MIDI")
    parser.add_argument('--length', type=int, default=1500)
    parser.add_argument('--output', default='output_generated.mid')
    args = parser.parse_args()

    melody_model = tf.keras.models.load_model(MELODY_MODEL_PATH)
    chord_model  = tf.keras.models.load_model(CHORD_MODEL_PATH)
    with open(SEED_PATH, 'rb') as f:
        seed = pickle.load(f)

    genres, moods, genre_enc, mood_enc = load_label_encoders()
    g_id = genre_enc.transform([genres[0]])[0]
    m_id = mood_enc.transform([moods[0]])[0]

    melody = generate_melody(melody_model, seed, g_id, m_id, length=args.length)
    chords = generate_chords(chord_model, melody.reshape(-1,1), g_id, m_id)

    midi_tools.note_state_matrix_to_midi(melody, chords, args.output)
    print(f"✔ Generation complete — MIDI saved as {args.output}")

if __name__ == "__main__":
    main()
