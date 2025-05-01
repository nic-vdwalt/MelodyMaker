import logging
import numpy as np
from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from typing import List, Optional

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(message)s')

def midi_to_note_state_matrix(
    midifile: str,
    lower_bound: int = 24,
    upper_bound: int = 102,
    squash: bool = False
) -> List[List[List[int]]]:
    """
    Convert a MIDI file into a note-state matrix.
    Each timestep is a list of [note_on, note_playing] for pitches in [lower_bound, upper_bound).
    """
    mid = MidiFile(midifile)
    resolution = mid.ticks_per_beat
    ticks_per_sample = max(1, resolution // 16)
    logger.debug(f"Loaded MIDI '{midifile}' resolution={resolution}, ticks_per_sample={ticks_per_sample}")

    tracks = mid.tracks
    # time until next message for each track, or None if finished
    timeleft = [trk[0].time if trk else None for trk in tracks]
    positions = [0] * len(tracks)

    span = upper_bound - lower_bound
    # [note_on, note_playing] per pitch
    state = [[0, 0] for _ in range(span)]
    statematrix: List[List[List[int]]] = []

    # step through in fixed-size ticks_per_sample slices
    while any(tl is not None for tl in timeleft):
        # process all messages whose time has elapsed in this slice
        for i, tl in enumerate(timeleft):
            # consume all zero-or-negative-time messages
            while tl is not None and tl <= 0:
                msg = tracks[i][positions[i]]
                if msg.type in ('note_on', 'note_off'):
                    pitch = msg.note
                    vel = getattr(msg, 'velocity', 0)
                    if lower_bound <= pitch < upper_bound:
                        idx = pitch - lower_bound
                        if msg.type == 'note_on' and vel > 0:
                            state[idx] = [1, 1]
                        else:
                            state[idx] = [0, 0]
                        # logger.debug(f"Raw msg: {msg.type} pitch={pitch} -> idx={idx}")
                    else:
                        logger.debug(f"Ignored msg: {msg.type} pitch={pitch} out of bounds")
                # advance to next message in this track
                positions[i] += 1
                if positions[i] < len(tracks[i]):
                    tl = tracks[i][positions[i]].time
                    timeleft[i] = tl
                else:
                    timeleft[i] = None
                    tl = None

            # subtract our slice duration
            if tl is not None:
                timeleft[i] -= ticks_per_sample

        # record a snapshot of the current state
        statematrix.append([row.copy() for row in state])

    logger.debug(f"Generated {len(statematrix)} frames")
    if squash:
        filtered = [f for i, f in enumerate(statematrix) if i == 0 or f != statematrix[i-1]]
        logger.debug(f"Squashed to {len(filtered)} frames")
        return filtered

    return statematrix

def note_state_matrix_to_midi(
    statematrix: List[List[List[int]]],
    chord_seq: Optional[List[int]],
    file_path: str,
    lower_bound: int = 24,
    upper_bound: int = 102
) -> None:
    """
    Convert either a full note-state matrix (t, span, 2) or a monophonic pitch list (t,) to a MIDI file.
    """
    span = upper_bound - lower_bound
    arr = np.array(statematrix)
    logger.debug(f"Input array shape: {arr.shape}")

    # If monophonic (1D), build a state matrix from pitch indices
    if arr.ndim == 1:
        pitches = arr.astype(int)
        mat_state = np.zeros((len(pitches), span, 2), dtype=int)
        for t, p in enumerate(pitches):
            if 0 <= p < span:
                mat_state[t, p] = [1, 1]
                logger.debug(f"Frame {t}: relative pitch {p} -> MIDI {p+lower_bound}")
            elif lower_bound <= p < upper_bound:
                idx = p - lower_bound
                mat_state[t, idx] = [1, 1]
                logger.debug(f"Frame {t}: absolute pitch {p} -> idx {idx}")
            else:
                logger.debug(f"Frame {t}: pitch {p} skipped (out of range)")
    elif arr.ndim == 3 and arr.shape[1:] == (span, 2):
        mat_state = arr.astype(int)
    else:
        raise ValueError(f"Unexpected input shape: {arr.shape}")

    length = mat_state.shape[0]
    # Safely prepare chords list without ambiguous truth-check
    if chord_seq is not None:
        chords = list(chord_seq)[:length]
    else:
        chords = [None] * length
    if len(chords) < length:
        chords += [None] * (length - len(chords))

    # Build MIDI
    midi = MidiFile(type=1)
    ticks_per_frame = midi.ticks_per_beat // 8
    meta = MidiTrack()
    meta.append(MetaMessage('set_tempo', tempo=bpm2tempo(120), time=0))
    meta.append(MetaMessage('end_of_track', time=0))
    mel = MidiTrack()
    mel.append(Message('program_change', program=0, channel=0, time=0))
    ch = MidiTrack()
    ch.append(Message('program_change', program=0, channel=1, time=0))
    midi.tracks.extend([meta, mel, ch])

    prev_state = np.zeros(span, dtype=int)
    prev_chord = None
    acc = {'mel': 0, 'ch': 0}

    for t in range(length):
        frame = mat_state[t]
        acc['mel'] += ticks_per_frame
        acc['ch'] += ticks_per_frame
        # melody off
        for i in range(span):
            if prev_state[i] == 1 and frame[i,1] == 0:
                mel.append(Message('note_off', note=i+lower_bound, velocity=0, time=acc['mel'], channel=0))
                acc['mel'] = 0
        # melody on
        for i in range(span):
            if prev_state[i] == 0 and frame[i,1] == 1:
                mel.append(Message('note_on', note=i+lower_bound, velocity=64, time=acc['mel'], channel=0))
                acc['mel'] = 0
        prev_state = frame[:,1].copy()

        # chord events
        chord = chords[t]
        if chord != prev_chord:
            if prev_chord is not None:
                for p in (prev_chord, prev_chord+4, prev_chord+7):
                    ch.append(Message('note_off', note=p, velocity=0, time=acc['ch'], channel=1))
                    acc['ch'] = 0
            if chord is not None:
                for p in (chord, chord+4, chord+7):
                    ch.append(Message('note_on', note=p, velocity=64, time=acc['ch'], channel=1))
                    acc['ch'] = 0
            prev_chord = chord

    # flush outs
    for i in range(span):
        if prev_state[i] == 1:
            mel.append(Message('note_off', note=i+lower_bound, velocity=0, time=acc['mel'], channel=0))
            acc['mel'] = 0
    if prev_chord is not None:
        for p in (prev_chord, prev_chord+4, prev_chord+7):
            ch.append(Message('note_off', note=p, velocity=0, time=acc['ch'], channel=1))
            acc['ch'] = 0
    mel.append(MetaMessage('end_of_track', time=0))
    ch.append(MetaMessage('end_of_track', time=0))

    midi.save(file_path)
    logger.info(f"Saved MIDI file to {file_path}")
