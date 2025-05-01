# File: src/midi_tools.py
import mido
import numpy as np
from typing import List, Optional


def midi_to_note_state_matrix(
    midifile: str,
    lower_bound: int = 24,
    upper_bound: int = 102,
    squash: bool = True
) -> List[List[List[int]]]:
    """
    Convert a MIDI file into a note-state matrix using mido.
    Each timestep is a list of [note_on, note_playing] for pitches in [lower_bound, upper_bound).
    If squash=True, consecutive identical frames are collapsed.
    """
    mid = mido.MidiFile(midifile)
    resolution = mid.ticks_per_beat
    ticks_per_sample = max(1, resolution // 8)

    tracks = mid.tracks
    timeleft = [trk[0].time if trk else None for trk in tracks]
    positions = [0] * len(tracks)

    span = upper_bound - lower_bound
    state = [[0, 0] for _ in range(span)]
    statematrix: List[List[List[int]]] = [[row.copy() for row in state]]

    time = 0
    while True:
        # process all tracks at this tick
        for i, trk in enumerate(tracks):
            if timeleft[i] is None:
                continue
            # consume all zero-delay messages
            while timeleft[i] == 0:
                msg = trk[positions[i]]
                if msg.type in ('note_on', 'note_off'):
                    p = msg.note
                    if lower_bound <= p < upper_bound:
                        idx = p - lower_bound
                        # note_off or note_on with velocity=0
                        if msg.type == 'note_off' or getattr(msg, 'velocity', 0) == 0:
                            state[idx] = [0, 0]
                        else:
                            state[idx] = [1, 1]
                elif msg.type == 'time_signature':
                    if msg.numerator not in (2, 4):
                        return statematrix  # unsupported time signature
                # advance in this track
                positions[i] += 1
                if positions[i] < len(trk):
                    timeleft[i] = trk[positions[i]].time
                else:
                    timeleft[i] = None
                    break
            # decrement time until next event
            if timeleft[i] is not None:
                timeleft[i] -= 1

        # stop when all tracks are done
        if all(t is None for t in timeleft):
            break

        time += 1
        if time % ticks_per_sample == 0:
            # prepare next frame: hold existing notes, reset 'new' flag
            new_state = [[on, 0] for on, _ in state]
            if not (squash and new_state == statematrix[-1]):
                statematrix.append([row.copy() for row in new_state])
            state = new_state

    return statematrix

def note_state_matrix_to_midi(
    statematrix: List[List[List[int]]],
    chord_seq: List[int],
    file_path: str,
    lower_bound: int = 24,
    upper_bound: int = 102,
    tick_scale: int = 55
) -> None:
    """
    Convert a note-state matrix or a monophonic pitch sequence, along with an underlying chord sequence, into a MIDI file.
    If `statematrix` is a list of pitch values (1D or shape (t,1)), it will be converted to a note-state matrix.
    Chord sequence entries are interpreted as MIDI root pitches, with major triads constructed for each timestep.

    Parameters:
    - statematrix: Either:
        • List of [ [on, new], … ] frames of length span = upper_bound−lower_bound, shape (t, span, 2)
        • List of pitch values per timestep, shape (t,) or (t,1)
    - chord_seq:   List of MIDI pitch numbers for chord roots per timestep
    - file_path:   Path where the .mid file will be saved
    - lower_bound: Lowest MIDI pitch (inclusive) used for melody
    - upper_bound: Highest MIDI pitch (exclusive) used for melody
    - tick_scale:  Multiplier for timing resolution (ticks per sample divided by 8)
    """
    lower_bound = int(lower_bound)
    upper_bound = int(upper_bound)
    span = upper_bound - lower_bound

    # Convert input to numpy array
    mat = np.array(statematrix)
    # Detect monophonic pitch sequence
    if mat.ndim == 1 or (mat.ndim == 2 and mat.shape[1] == 1):
        pitches = mat.flatten().astype(int)
        # build full note-state matrix of zeros
        full = np.zeros((len(pitches), span, 2), dtype=int)
        for t, p in enumerate(pitches):
            if lower_bound <= p < upper_bound:
                idx = p - lower_bound
                full[t, idx] = [1, 1]
        mat = full
    # Ensure shape (t, span, 2)
    elif mat.ndim == 2 and mat.shape[1] == span * 2:
        mat = mat.reshape(-1, span, 2)
    elif mat.ndim == 3 and mat.shape[1:] == (span, 2):
        pass
    else:
        raise ValueError(f"Unsupported statematrix shape {mat.shape}; expected pitch list or (t,{span},2)")

    # Create MIDI with melody and chord tracks
    mid = mido.MidiFile()
    mid.ticks_per_beat = tick_scale * 8
    melody_track = mido.MidiTrack()
    chord_track = mido.MidiTrack()
    mid.tracks.extend([melody_track, chord_track])

    last_m_time = 0
    last_c_time = 0
    prev_m = [[0, 0] for _ in range(span)]
    prev_chord = None
    active_chords: List[int] = []

    for t, frame in enumerate(mat):
        # Chord handling
        cur_ch = chord_seq[t] if t < len(chord_seq) else None
        if cur_ch != prev_chord:
            # Turn off old chord notes
            for note in active_chords:
                delta = (t - last_c_time) * tick_scale
                chord_track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
                last_c_time = t
            active_chords = []
            # Turn on new chord triad
            if cur_ch is not None:
                triad = [cur_ch, cur_ch + 4, cur_ch + 7]
                for note in triad:
                    delta = (t - last_c_time) * tick_scale
                    chord_track.append(mido.Message('note_on', note=note, velocity=40, time=delta))
                    last_c_time = t
                active_chords = triad
            prev_chord = cur_ch

        # Melody handling
        offs = [i for i, (c, p) in enumerate(zip(frame, prev_m)) if p[0] and not c[0]]
        ons  = [i for i, (c, p) in enumerate(zip(frame, prev_m)) if c[0] and not p[0]]
        for idx in offs:
            delta = (t - last_m_time) * tick_scale
            melody_track.append(mido.Message('note_off', note=idx + lower_bound, velocity=0, time=delta))
            last_m_time = t
        for idx in ons:
            delta = (t - last_m_time) * tick_scale
            melody_track.append(mido.Message('note_on', note=idx + lower_bound, velocity=40, time=delta))
            last_m_time = t
        prev_m = [row.copy() for row in frame]

    # End remaining notes
    for note in active_chords:
        chord_track.append(mido.Message('note_off', note=note, velocity=0, time=tick_scale))
    melody_track.append(mido.MetaMessage('end_of_track', time=tick_scale))
    chord_track.append(mido.MetaMessage('end_of_track', time=0))

    mid.save(file_path)