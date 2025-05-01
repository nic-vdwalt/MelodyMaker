# config.py
import os

"""
Configuration constants for the MelodyMaker project
"""

MIDI_DIR = 'Rock_Music_Midi'
LOWER_BOUND = 24
UPPER_BOUND = 102
SEQ_LEN = 60
EMBED_DIM = 16
LSTM_UNITS = 128
DROPOUT = 0.3
BATCH_SIZE = 64
EPOCHS = 20
SEED_PATH = 'seed.pkl'
MELODY_MODEL_PATH = 'melody_model.h5'
CHORD_MODEL_PATH = 'chord_model.h5'
NUM_CHORDS = 12