# generate.py

import argparse
import pickle
import tensorflow as tf
from utils.key_utils import determine_key_from_phrase
from utils.midi_utils import note_state_matrix_to_midi
from utils.model_utils import generate_melody, generate_chords
from config import SEED_PATH, MELODY_MODEL_PATH, CHORD_MODEL_PATH

def main():
    parser = argparse.ArgumentParser(description="Generate melody and chord MIDI")
    parser.add_argument('--length', type=int, default=1500)
    parser.add_argument('--output', default='output_generated.mid')
    parser.add_argument(
        '--phrase',
        default="Your prompt here",
        help="Text prompt whose first letter maps to a key signature"
    )
    args = parser.parse_args()

    melody_model = tf.keras.models.load_model(MELODY_MODEL_PATH)
    chord_model  = tf.keras.models.load_model(CHORD_MODEL_PATH)

    with open(SEED_PATH, 'rb') as f:
        seed = pickle.load(f)

    key_id = determine_key_from_phrase(args.phrase)

    melody = generate_melody(melody_model, seed, key_id, length=args.length)
    chords = generate_chords(chord_model, melody.reshape(-1,1), key_id)

    note_state_matrix_to_midi(melody, chords, args.output)
    print(f"✔ Generation complete — MIDI saved as {args.output}")

if __name__ == "__main__":
    main()
