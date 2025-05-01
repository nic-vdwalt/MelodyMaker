# train.py
import argparse
import pickle
import tensorflow as tf
from utils.data_utils import load_label_encoders, prepare_training_data
from utils.model_utils import build_melody_model, build_chord_model
import report as r
from config import MIDI_DIR, SEED_PATH, MELODY_MODEL_PATH, CHORD_MODEL_PATH, BATCH_SIZE, EPOCHS

def train(midi_dir, seed_path, melody_path, chord_path, batch_size, epochs):
    genres, moods, genre_enc, mood_enc = load_label_encoders()
    r.updateData(genres, moods)

    X_notes, y_notes, X_genre, X_mood = prepare_training_data(
        midi_dir, genres, moods, genre_enc, mood_enc
    )

    melody_model = build_melody_model(len(genres), len(moods))
    chord_model  = build_chord_model(len(genres), len(moods))

    melody_model.fit(
        [X_notes, X_genre, X_mood], y_notes,
        batch_size=batch_size, epochs=epochs,
        validation_split=0.1, shuffle=True
    )
    melody_model.save(melody_path)
    chord_model.save(chord_path)

    with open(seed_path, 'wb') as f:
        pickle.dump(X_notes[0], f)

    print("✔ Training complete — models and seed saved.")


def main():
    parser = argparse.ArgumentParser(description="Train melody and chord models")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs',     type=int, default=EPOCHS)
    parser.add_argument('--midi_dir',   default=MIDI_DIR)
    parser.add_argument('--seed_path',  default=SEED_PATH)
    parser.add_argument('--melody_path',default=MELODY_MODEL_PATH)
    parser.add_argument('--chord_path', default=CHORD_MODEL_PATH)
    args = parser.parse_args()
    train(
        args.midi_dir, args.seed_path,
        args.melody_path, args.chord_path,
        args.batch_size, args.epochs
    )

if __name__ == "__main__":
    main()
