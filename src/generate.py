# generate.py

import argparse
import pickle
import tensorflow as tf
from utils.event_utils import generate_events, event_sequence_to_midi
from config import SEED_PATH, MELODY_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Generate MIDI via event-based model"
    )
    parser.add_argument(
        '--length',
        type=int,
        default=1024,
        help="Number of events to generate"
    )
    parser.add_argument(
        '--output',
        default='output_generated.mid',
        help="Path to output MIDI file"
    )
    args = parser.parse_args()

    # Load a seed event sequence (list of token indices)
    with open(SEED_PATH, 'rb') as f:
        seed_events = pickle.load(f)

    # Load the trained event-based model
    model = tf.keras.models.load_model(MELODY_MODEL_PATH)

    # Auto-regressively sample an event sequence and write to MIDI
    events = generate_events(model, seed_events, length=args.length)
    event_sequence_to_midi(events, args.output)

    print(f"✔ Generation complete — MIDI saved as {args.output}")


if __name__ == "__main__":
    main()