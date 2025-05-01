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
    Convert a note-state matrix and an underlying chord sequence into a MIDI file.
    Chord sequence entries are interpreted as MIDI root pitches, with major triads constructed.

    Parameters:
    - statematrix: List of [ [on, new], … ] frames of length span = upper_bound-lower_bound
    - chord_seq:   List of MIDI pitch numbers for chord roots
    - file_path:   Path where the .mid file will be saved
    - lower_bound: Lowest MIDI pitch (inclusive) for melody
    - upper_bound: Highest MIDI pitch (exclusive) for melody
    - tick_scale:  Multiplier for timing resolution
    """
    # normalize bounds
    lower_bound = int(lower_bound)
    upper_bound = int(upper_bound)
    span = upper_bound - lower_bound

    # prepare state matrix
    mat = np.array(statematrix)
    if mat.ndim == 2:
        mat = mat.reshape(-1, span * 2)
        mat = np.dstack((mat[:, :span], mat[:, span:]))

    # create MIDI file with two tracks: melody and chords
    mid = mido.MidiFile()
    mid.ticks_per_beat = tick_scale * 8
    melody_track = mido.MidiTrack()
    chord_track = mido.MidiTrack()
    mid.tracks.append(melody_track)
    mid.tracks.append(chord_track)

    # initialize timing and previous states
    last_melody_time = 0
    last_chord_time = 0
    prev_melody = [[0, 0] for _ in range(span)]
    prev_chord = None
    active_chord_notes: List[int] = []

    # iterate through timesteps
    for t, frame in enumerate(mat):
        # handle chord changes
        cur_chord = chord_seq[t] if t < len(chord_seq) else None
        if cur_chord != prev_chord:
            # turn off previous chord notes
            for note in active_chord_notes:
                delta = (t - last_chord_time) * tick_scale
                chord_track.append(
                    mido.Message('note_off', note=note, velocity=0, time=delta)
                )
                last_chord_time = t

            active_chord_notes = []
            if cur_chord is not None:
                # build a major triad: root, major third, perfect fifth
                triad = [cur_chord, cur_chord + 4, cur_chord + 7]
                for note in triad:
                    delta = (t - last_chord_time) * tick_scale
                    chord_track.append(
                        mido.Message('note_on', note=note, velocity=40, time=delta)
                    )
                    last_chord_time = t
                active_chord_notes = triad

            prev_chord = cur_chord

        # handle melody note events
        offs = [i for i, (cur, pr) in enumerate(zip(frame, prev_melody)) if pr[0] == 1 and cur[0] == 0]
        ons  = [i for i, (cur, pr) in enumerate(zip(frame, prev_melody)) if cur[0] == 1 and pr[0] == 0]

        for note_idx in offs:
            delta = (t - last_melody_time) * tick_scale
            melody_track.append(
                mido.Message('note_off', note=note_idx + lower_bound, velocity=0, time=delta)
            )
            last_melody_time = t
        for note_idx in ons:
            delta = (t - last_melody_time) * tick_scale
            melody_track.append(
                mido.Message('note_on', note=note_idx + lower_bound, velocity=40, time=delta)
            )
            last_melody_time = t

        prev_melody = [row.copy() for row in frame]

    # end any remaining chord notes
    for note in active_chord_notes:
        chord_track.append(
            mido.Message('note_off', note=note, velocity=0, time=tick_scale)
        )
    # end tracks
    melody_track.append(mido.MetaMessage('end_of_track', time=tick_scale))
    chord_track.append(mido.MetaMessage('end_of_track', time=0))

    # save MIDI file
    mid.save(file_path)

    """
    Convert a note-state matrix (and optional chord sequence) into a MIDI file using mido.
    
    Parameters:
    - statematrix: List of [ [on, new], … ] frames of length span = upper_bound-lower_bound
    - chord_seq:   List of chord indices (currently unused)
    - file_path:   Path where the .mid file will be saved
    - lower_bound: Lowest MIDI pitch (inclusive)
    - upper_bound: Highest MIDI pitch (exclusive)
    - tick_scale:  Multiplier for timing resolution
    """
    # ensure bounds are integers
    lower_bound = int(lower_bound)
    upper_bound = int(upper_bound)
    span = upper_bound - lower_bound

    # prepare the matrix
    mat = np.array(statematrix)
    if mat.ndim == 2:
        # reshape flattened frames back into [timesteps, span, 2]
        mat = mat.reshape(-1, span * 2)
        mat = np.dstack((mat[:, :span], mat[:, span:]))

    mid = mido.MidiFile()
    mid.ticks_per_beat = tick_scale * 8
    track = mido.MidiTrack()
    mid.tracks.append(track)

    last_time = 0
    prev = [[0, 0] for _ in range(span)]

    for t, frame in enumerate(mat):
        # notes that stopped
        offs = [i for i, (cur, pr) in enumerate(zip(frame, prev)) if pr[0] == 1 and cur[0] == 0]
        # notes that started (or are newly held)
        ons  = [i for i, (cur, pr) in enumerate(zip(frame, prev)) if cur[0] == 1 and (pr[0] == 0 or cur[1] == 1)]

        for note in offs:
            delta = (t - last_time) * tick_scale
            track.append(
                mido.Message('note_off', note=note + lower_bound, velocity=0, time=delta)
            )
            last_time = t
        for note in ons:
            delta = (t - last_time) * tick_scale
            track.append(
                mido.Message('note_on', note=note + lower_bound, velocity=40, time=delta)
            )
            last_time = t

        prev = [row.copy() for row in frame]

    # end of track
    track.append(mido.MetaMessage('end_of_track', time=tick_scale))
    mid.save(file_path)