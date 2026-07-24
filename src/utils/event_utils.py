import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from typing import List

# --- Event Vocabulary Generators ---
NOTE_ON    = lambda p: f"note_on_{p}"
NOTE_OFF   = lambda p: f"note_off_{p}"
TIME_SHIFT = lambda dt: f"time_shift_{dt}"
BEAT_POS   = lambda pos: f"beat_pos_{pos}"
VEL        = lambda v: f"vel_{v}"

# Constants
PITCH_MIN        = 24
PITCH_MAX        = 102
TIME_BINS        = list(range(1, 33))       # quantized ticks (1–32)
BEAT_SUBDIV      = 16                       # 16th-note positions in a bar
VELOCITY_BUCKETS = [20, 40, 60, 80, 100, 120]

# Build full vocabulary list
VOCAB = []
for pitch in range(PITCH_MIN, PITCH_MAX):
    VOCAB += [NOTE_ON(pitch), NOTE_OFF(pitch)]
VOCAB += [TIME_SHIFT(dt) for dt in TIME_BINS]
VOCAB += [BEAT_POS(pos) for pos in range(BEAT_SUBDIV)]
VOCAB += [VEL(v) for v in VELOCITY_BUCKETS]

# Mapping dictionaries
EVENT2IDX = {event: idx for idx, event in enumerate(VOCAB)}
IDX2EVENT = {idx: event for event, idx in EVENT2IDX.items()}


def _velocity_bucket(velocity: int) -> int:
    return min(VELOCITY_BUCKETS, key=lambda bucket: abs(bucket - velocity))


def midi_to_event_sequence(file_path: str) -> List[int]:
    midi = MidiFile(file_path)
    ticks_per_bin = max(1, midi.ticks_per_beat // BEAT_SUBDIV)
    events = []
    elapsed_ticks = 0
    velocity = None

    for message in MidiFile(file_path).merged_track:
        elapsed_ticks += message.time
        is_note_on = message.type == 'note_on' and message.velocity > 0
        is_note_off = message.type == 'note_off' or (
            message.type == 'note_on' and message.velocity == 0
        )
        if not is_note_on and not is_note_off:
            continue
        if message.note < PITCH_MIN or message.note >= PITCH_MAX:
            continue

        bins = max(0, round(elapsed_ticks / ticks_per_bin))
        while bins > 0:
            step = min(bins, TIME_BINS[-1])
            events.append(EVENT2IDX[TIME_SHIFT(step)])
            bins -= step
        elapsed_ticks = 0

        beat_pos = int(message.time // ticks_per_bin) % BEAT_SUBDIV
        events.append(EVENT2IDX[BEAT_POS(beat_pos)])
        if is_note_on:
            bucket = _velocity_bucket(message.velocity)
            if bucket != velocity:
                events.append(EVENT2IDX[VEL(bucket)])
                velocity = bucket
            events.append(EVENT2IDX[NOTE_ON(message.note)])
        else:
            events.append(EVENT2IDX[NOTE_OFF(message.note)])

    return events


def generate_events(
    model,
    seed_events: List[int],
    length: int = 1024,
    temperature: float = 1.0
) -> List[int]:
    """
    Auto-regressively sample a sequence of event-token indices from the model.
    """
    seq = list(seed_events)
    for _ in range(length):
        x = np.array([seq], dtype='int32')         # shape (1, T)
        logits = model.predict(x, verbose=0)[0, -1]  # logits over VOCAB

        # Apply temperature
        logits = np.log(np.clip(logits, 1e-8, None)) / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        next_idx = np.random.choice(len(probs), p=probs)
        seq.append(int(next_idx))
    return seq


def event_sequence_to_midi(
    event_seq: List[int],
    file_path: str,
    tempo_bpm: int = 120
) -> None:
    """
    Convert an event-token sequence back into a MIDI file.
    """
    midi = MidiFile(type=1)
    meta = MidiTrack()
    mel  = MidiTrack()
    midi.tracks.extend([meta, mel])

    meta.append(MetaMessage('set_tempo', tempo=bpm2tempo(tempo_bpm), time=0))
    ticks_per_bin = midi.ticks_per_beat // BEAT_SUBDIV
    time_acc = 0
    current_vel = 64

    for idx in event_seq:
        evt = IDX2EVENT[idx]
        if evt.startswith('time_shift_'):
            # accumulate delta time
            dt = int(evt.split('_')[2])
            time_acc += dt * ticks_per_bin
        elif evt.startswith('beat_pos_'):
            # beat position feature (no direct MIDI action)
            continue
        elif evt.startswith('vel_'):
            # update current velocity bucket
            current_vel = int(evt.split('_')[1])
        elif evt.startswith('note_on_'):
            pitch = int(evt.split('_')[2])
            mel.append(Message(
                'note_on', note=pitch, velocity=current_vel, time=time_acc
            ))
            time_acc = 0
        elif evt.startswith('note_off_'):
            pitch = int(evt.split('_')[2])
            mel.append(Message(
                'note_off', note=pitch, velocity=0, time=time_acc
            ))
            time_acc = 0

    # End-of-track
    mel.append(MetaMessage('end_of_track', time=1))
    meta.append(MetaMessage('end_of_track', time=0))
    midi.save(file_path)