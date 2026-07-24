import argparse
import glob
import json
import os
import pickle
import tempfile
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from config import BATCH_SIZE, EPOCHS, MELODY_MODEL_PATH, MIDI_DIR, SEED_PATH, SEQ_LEN
from utils.event_model import build_event_model
from utils.event_utils import EVENT2IDX, midi_to_event_sequence


def midi_files(midi_dir):
    patterns = [os.path.join(midi_dir, '**', '*.mid'), os.path.join(midi_dir, '**', '*.midi')]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(path for path in files if not os.path.basename(path).startswith('._'))


def scan_event_dataset(midi_dir, seq_len, on_file=None):
    files = midi_files(midi_dir)
    if not files:
        raise ValueError(f"No MIDI files found in {midi_dir}")

    valid_files = []
    skipped_files = []
    total_windows = 0
    first_seed = None
    for index, path in enumerate(files):
        try:
            sequence = midi_to_event_sequence(path)
        except Exception as error:
            skipped_files.append((path, str(error)))
            if on_file:
                on_file(index + 1, len(files), path, total_windows, str(error))
            continue
        windows = max(0, len(sequence) - seq_len)
        if windows:
            valid_files.append(path)
            total_windows += windows
            if first_seed is None:
                first_seed = sequence[:seq_len]
        if on_file:
            on_file(index + 1, len(files), path, total_windows, None)

    if total_windows == 0:
        raise ValueError(f"No event sequences longer than {seq_len} found in {midi_dir}")
    return valid_files, skipped_files, total_windows, first_seed


def event_window_generator(files, seq_len):
    for path in files:
        sequence = midi_to_event_sequence(path)
        for index in range(len(sequence) - seq_len):
            window = np.asarray(sequence[index:index + seq_len + 1], dtype='int32')
            yield window[:-1], window[1:]


def make_dataset(files, seq_len, batch_size, total_windows):
    signature = (
        tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(seq_len,), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: event_window_generator(files, seq_len), output_signature=signature
    )
    return dataset.shuffle(min(total_windows, 10000), seed=42).batch(batch_size).prefetch(2)


class StatusWriter(tf.keras.callbacks.Callback):
    def __init__(self, path, total_epochs):
        super().__init__()
        self.path = path
        self.total_epochs = total_epochs
        self.started_at = time.time()

    def write(self, phase, **values):
        if not self.path:
            return
        payload = {
            'phase': phase,
            'updated_at': time.time(),
            'elapsed_seconds': time.time() - self.started_at,
            **values,
        }
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(prefix='.melody-status-', dir=directory)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as status_file:
                json.dump(payload, status_file)
            os.replace(temporary_path, self.path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def on_train_begin(self, logs=None):
        self.write('training', epoch=0, epochs=self.total_epochs, progress=0.0)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        completed = epoch + 1
        self.write(
            'training',
            epoch=completed,
            epochs=self.total_epochs,
            progress=completed / self.total_epochs,
            loss=float(logs.get('loss', 0.0)),
            val_loss=float(logs.get('val_loss', 0.0)),
        )


def main():
    parser = argparse.ArgumentParser(description="Train the event-based Melody model")
    parser.add_argument('--midi_dir', default=MIDI_DIR)
    parser.add_argument('--seed_path', default=SEED_PATH)
    parser.add_argument('--melody_path', default=MELODY_MODEL_PATH)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--seq_len', type=int, default=SEQ_LEN)
    parser.add_argument('--status_path', default='')
    args = parser.parse_args()

    if args.batch_size < 1 or args.epochs < 1 or args.seq_len < 1:
        parser.error('batch size, epochs, and sequence length must be positive')

    writer = StatusWriter(args.status_path, args.epochs)
    writer.write('scanning', files_scanned=0, files_total=len(midi_files(args.midi_dir)), progress=0.0)

    def on_file(index, total, path, windows, error):
        writer.write(
            'scanning',
            files_scanned=index,
            files_total=total,
            current_file=os.path.basename(path),
            windows=windows,
            skipped=bool(error),
            progress=index / total,
        )

    try:
        files, skipped, total_windows, seed = scan_event_dataset(
            args.midi_dir, args.seq_len, on_file
        )
        print(f"Dataset: {len(files)} usable MIDI files, {total_windows} windows")
        if skipped:
            print(f"Skipped {len(skipped)} unreadable MIDI files")

        validation_windows = max(1, total_windows // 10)
        training_windows = total_windows - validation_windows
        dataset = make_dataset(files, args.seq_len, args.batch_size, total_windows)
        validation = dataset.take(validation_windows // args.batch_size + 1)
        training = dataset.skip(validation_windows // args.batch_size + 1)

        tf.random.set_seed(42)
        model = build_event_model(len(EVENT2IDX))
        model.compile(optimizer='adam', loss=SparseCategoricalCrossentropy(from_logits=True))
        model.summary()
        model.fit(training, validation_data=validation, epochs=args.epochs, callbacks=[writer])

        model.save(args.melody_path)
        with open(args.seed_path, 'wb') as seed_file:
            pickle.dump(seed, seed_file)
        writer.write(
            'complete',
            epoch=args.epochs,
            epochs=args.epochs,
            progress=1.0,
            model_path=os.path.abspath(args.melody_path),
            seed_path=os.path.abspath(args.seed_path),
        )
        print("Training complete — model and seed saved.")
    except Exception as error:
        writer.write('error', message=str(error), progress=0.0)
        raise


if __name__ == "__main__":
    main()
