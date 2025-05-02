import argparse
import os
import glob
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import MIDI_DIR, SEED_PATH, MELODY_MODEL_PATH, BATCH_SIZE, EPOCHS, SEQ_LEN
from utils.event_utils import midi_to_event_sequence, EVENT2IDX
from utils.event_model import build_event_model


def load_event_dataset(midi_dir, seq_len):
    all_seqs = []
    for f in glob.glob(os.path.join(midi_dir, '*.mid')):
        seq = midi_to_event_sequence(f)
        if len(seq) > seq_len:
            for i in range(len(seq) - seq_len):
                window = seq[i:i + seq_len + 1]
                all_seqs.append(window)
    if not all_seqs:
        raise ValueError(f"No event sequences longer than {seq_len} found in {midi_dir}")

    data = np.array(all_seqs, dtype='int32')
    X = data[:, :-1]
    y = data[:, 1:]
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Train event-based transformer model")
    parser.add_argument('--midi_dir',   default=MIDI_DIR)
    parser.add_argument('--seed_path',  default=SEED_PATH)
    parser.add_argument('--melody_path',default=MELODY_MODEL_PATH)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs',     type=int, default=EPOCHS)
    parser.add_argument('--seq_len',    type=int, default=SEQ_LEN,
                        help="Sliding window length for event sequences")
    args = parser.parse_args()

    tf.random.set_seed(42)
    X, y = load_event_dataset(args.midi_dir, args.seq_len)
    vocab_size = len(EVENT2IDX)

    model = build_event_model(vocab_size)
    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
    )

    model.summary()
    model.fit(
        X, y,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=0.1,
        shuffle=True
    )
    model.save(args.melody_path)

    # Save the first input window as the seed sequence
    with open(args.seed_path, 'wb') as f:
        pickle.dump(list(X[0]), f)

    print("✔ Training complete — model and seed saved.")

if __name__ == "__main__":
    main()
