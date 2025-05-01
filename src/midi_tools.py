import mido
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from typing import List, Optional


def midi_to_note_state_matrix(
    midifile: str,
    lower_bound: int = 24,
    upper_bound: int = 102,
    squash: bool = True
) -> List[List[List[int]]]:
    """
    Convert a MIDI file into a note-state matrix.
    Each timestep is a list of [note_on, note_playing] for pitches in [lower_bound, upper_bound).
    If squash=True, consecutive identical frames are collapsed.

    Returns:
        statematrix: List of frames; each frame is a list of [on, playing] per pitch.
    """
    mid = MidiFile(midifile)
    resolution = mid.ticks_per_beat
    ticks_per_sample = max(1, resolution // 8)

    span = upper_bound - lower_bound
    state = [[0, 0] for _ in range(span)]
    statematrix: List[List[List[int]]] = []

    # Track positions and time left
    tracks = mid.tracks
    time_left = [trk[0].time if trk else None for trk in tracks]
    positions = [0] * len(tracks)
    current_time = 0

    while True:
        # find next event
        valid = [t for t in time_left if t is not None]
        if not valid:
            break
        min_time = min(valid)
        current_time += min_time

        # advance all tracks
        for i, t in enumerate(time_left):
            if t is None:
                continue
            if t == min_time:
                msg = tracks[i][positions[i]]
                # handle note events
                if not msg.is_meta and msg.type in ('note_on', 'note_off'):
                    pitch = msg.note
                    if lower_bound <= pitch < upper_bound:
                        idx = pitch - lower_bound
                        if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                            state[idx] = [1, 1]
                        else:
                            state[idx] = [0, 0]
                positions[i] += 1
                # update next time
                if positions[i] < len(tracks[i]):
                    time_left[i] = tracks[i][positions[i]].time
                else:
                    time_left[i] = None
            else:
                time_left[i] -= min_time

        # record a frame when enough ticks have passed
        if current_time >= ticks_per_sample:
            statematrix.append([row.copy() for row in state])
            current_time %= ticks_per_sample

    # optionally squash consecutive duplicates
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
    upper_bound: int = 102,
    tick_scale: int = 55
) -> None:
    """
    Convert a note-state matrix (or monophonic pitch sequence) and optional chord roots
    into a MIDI file saved at `file_path`.

    - statematrix: either:
        • list of [on, playing] frames, shape (t, span, 2)
        • list/array of pitch values, shape (t,) or (t, 1)
    - chord_seq:  list of MIDI pitch roots per timestep (or None)
    """
    # ensure numpy array
    mat = np.array(statematrix)
    span = upper_bound - lower_bound

    # detect monophonic sequence
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
        raise ValueError(f"Expected statematrix shape (t,{span},2) or (t,) or (t,1); got {mat.shape}")

    # prepare chord sequence
    if chord_seq is None:
        chord_seq = [None] * mat_state.shape[0]
    if len(chord_seq) < mat_state.shape[0]:
        chord_seq = list(chord_seq) + [None] * (mat_state.shape[0] - len(chord_seq))

    # create MIDI
    midi = MidiFile()
    # tempo track
    meta = MidiTrack()
    meta.append(MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0))
    midi.tracks.append(meta)

    # melody track (channel 0)
    mel_track = MidiTrack()
    midi.tracks.append(mel_track)
    # chord track (channel 1)
    ch_track = MidiTrack()
    midi.tracks.append(ch_track)

    prev_state = np.zeros(span, dtype=int)
    prev_chord = None
    time_acc = 0

    for t, frame in enumerate(mat_state):
        # melody on/off
        for i in range(span):
            on, playing = frame[i]
            if playing and not prev_state[i]:
                mel_track.append(Message('note_on', note=i+lower_bound, velocity=64,
                                         time=time_acc, channel=0))
                time_acc = 0
            if not playing and prev_state[i]:
                mel_track.append(Message('note_off', note=i+lower_bound, velocity=64,
                                         time=time_acc, channel=0))
                time_acc = 0
        prev_state = frame[:, 1].copy()

        # chord on/off
        chord = chord_seq[t]
        if chord is not None:
            if prev_chord is None:
                # first chord
                notes = [chord, chord+4, chord+7]
                for p in notes:
                    ch_track.append(Message('note_on', note=p, velocity=64,
                                             time=time_acc, channel=1))
                    time_acc = 0
                prev_chord = chord
            elif chord != prev_chord:
                # chord change: turn off old
                old_notes = [prev_chord, prev_chord+4, prev_chord+7]
                for p in old_notes:
                    ch_track.append(Message('note_off', note=p, velocity=64,
                                             time=time_acc, channel=1))
                    time_acc = 0
                # new chord on
                new_notes = [chord, chord+4, chord+7]
                for p in new_notes:
                    ch_track.append(Message('note_on', note=p, velocity=64,
                                             time=time_acc, channel=1))
                    time_acc = 0
                prev_chord = chord

        # advance time
        time_acc += tick_scale

    # finalize: turn off sustained notes
    for i in range(span):
        if prev_state[i]:
            mel_track.append(Message('note_off', note=i+lower_bound, velocity=64,
                                     time=time_acc, channel=0))
            time_acc = 0
    if prev_chord is not None:
        final_notes = [prev_chord, prev_chord+4, prev_chord+7]
        for p in final_notes:
            ch_track.append(Message('note_off', note=p, velocity=64,
                                     time=time_acc, channel=1))
            time_acc = 0

    midi.save(file_path)
