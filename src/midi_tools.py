import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from typing import List, Optional


def midi_to_note_state_matrix(
    midifile: str,
    lower_bound: int = 24,
    upper_bound: int = 102,
    squash: bool = False
) -> List[List[List[int]]]:
    """
    Convert a MIDI file into a note-state matrix.
    Each timestep is a list of [note_on, note_playing] for pitches in [lower_bound, upper_bound).
    squash=False preserves all frames to maintain original duration.

    Returns:
        statematrix: List of frames; each frame is a list of [on, playing] per pitch.
    """
    mid = MidiFile(midifile)
    resolution = mid.ticks_per_beat
    ticks_per_sample = max(1, resolution // 8)

    span = upper_bound - lower_bound
    state = [[0, 0] for _ in range(span)]
    statematrix: List[List[List[int]]] = []

    tracks = mid.tracks
    time_left = [trk[0].time if trk else None for trk in tracks]
    positions = [0] * len(tracks)
    current_time = 0

    while True:
        valid = [t for t in time_left if t is not None]
        if not valid:
            break
        min_time = min(valid)
        current_time += min_time

        for i, t in enumerate(time_left):
            if t is None:
                continue
            if t == min_time:
                msg = tracks[i][positions[i]]
                if not msg.is_meta and msg.type in ('note_on', 'note_off'):
                    pitch = msg.note
                    if lower_bound <= pitch < upper_bound:
                        idx = pitch - lower_bound
                        if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                            state[idx] = [1, 1]
                        else:
                            state[idx] = [0, 0]
                positions[i] += 1
                if positions[i] < len(tracks[i]):
                    time_left[i] = tracks[i][positions[i]].time
                else:
                    time_left[i] = None
            else:
                time_left[i] -= min_time

        if current_time >= ticks_per_sample:
            statematrix.append([row.copy() for row in state])
            current_time %= ticks_per_sample

    if squash:
        squashed = []
        prev = None
        for frame in statematrix:
            if frame != prev:
                squashed.append(frame)
            prev = frame
        return squashed
    return statematrix


def note_state_matrix_to_midi(
    statematrix: List[List[List[int]]],
    chord_seq: Optional[List[int]],
    file_path: str,
    lower_bound: int = 24,
    upper_bound: int = 102
) -> None:
    """
    Convert a note-state matrix (or mono pitch list) + optional chord roots into a MIDI file.
    Uses accumulated time deltas to preserve full duration and ensure melody/bass tracks play correctly.
    """
    span = upper_bound - lower_bound
    mat = np.array(statematrix)

    # Expand mono lists to full state matrix
    if mat.ndim == 1 or (mat.ndim == 2 and mat.shape[1] == 1):
        pitches = mat.flatten().astype(int)
        mat_state = np.zeros((len(pitches), span, 2), dtype=int)
        for t, p in enumerate(pitches):
            idx = p - lower_bound
            if 0 <= idx < span:
                mat_state[t, idx] = [1, 1]
    elif mat.ndim == 3 and mat.shape[1] == span and mat.shape[2] == 2:
        mat_state = mat.astype(int)
    else:
        raise ValueError(f"Expected statematrix shape (t,{span},2) or (t,) or (t,1); got {mat.shape})")

    # Prepare chords
    length = mat_state.shape[0]
    if chord_seq is None:
        chord_seq = [None] * length
    chord_seq = list(chord_seq) + [None] * max(0, length - len(chord_seq))

    # Create MIDI file with default ticks_per_beat
    midi = MidiFile(type=1)
    ticks_per_frame = midi.ticks_per_beat // 8

    # Tempo track
    meta = MidiTrack()
    meta.append(MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0))
    meta.append(MetaMessage('end_of_track', time=0))
    midi.tracks.append(meta)

    # Melody track
    mel = MidiTrack()
    mel.append(Message('program_change', program=0, channel=0, time=0))
    midi.tracks.append(mel)

    # Chord track
    ch = MidiTrack()
    ch.append(Message('program_change', program=0, channel=1, time=0))
    midi.tracks.append(ch)

    prev_state = np.zeros(span, dtype=int)
    prev_chord = None
    mel_time_acc = 0
    ch_time_acc = 0

    for t, frame in enumerate(mat_state):
        mel_time_acc += ticks_per_frame
        ch_time_acc += ticks_per_frame

        # Melody note_off events
        for i in range(span):
            if prev_state[i] == 1 and frame[i, 1] == 0:
                mel.append(Message('note_off', note=i + lower_bound, velocity=0,
                                   time=mel_time_acc, channel=0))
                mel_time_acc = 0
        # Melody note_on events
        for i in range(span):
            if prev_state[i] == 0 and frame[i, 1] == 1:
                mel.append(Message('note_on', note=i + lower_bound, velocity=64,
                                   time=mel_time_acc, channel=0))
                mel_time_acc = 0
        prev_state = frame[:, 1].copy()

        # Chord events
        chord = chord_seq[t]
        if chord != prev_chord:
            # turn off old chord
            if prev_chord is not None:
                for p in (prev_chord, prev_chord + 4, prev_chord + 7):
                    ch.append(Message('note_off', note=p, velocity=0,
                                       time=ch_time_acc, channel=1))
                    ch_time_acc = 0
            # turn on new chord
            if chord is not None:
                for p in (chord, chord + 4, chord + 7):
                    ch.append(Message('note_on', note=p, velocity=64,
                                       time=ch_time_acc, channel=1))
                    ch_time_acc = 0
            prev_chord = chord

    # Final note_offs for melody
    for i in range(span):
        if prev_state[i]:
            mel.append(Message('note_off', note=i + lower_bound, velocity=0,
                               time=mel_time_acc, channel=0))
            mel_time_acc = 0
    # Final note_offs for chords
    if prev_chord is not None:
        for p in (prev_chord, prev_chord + 4, prev_chord + 7):
            ch.append(Message('note_off', note=p, velocity=0,
                               time=ch_time_acc, channel=1))
            ch_time_acc = 0

    # End-of-track markers
    mel.append(MetaMessage('end_of_track', time=0))
    ch.append(MetaMessage('end_of_track', time=0))

    midi.save(file_path)
