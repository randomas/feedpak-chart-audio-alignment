"""
feedpak_common.py

Shared, dependency-light helpers used by all three pipeline scripts
(process_vocals.py, process_gp_alignment.py, build_feedpak.py).

Kept free of heavy imports (guitarpro, librosa, whisperx, torchcrepe) so it
can be imported and unit-tested without those installed.
"""
import json
import hashlib
import os
import math
import re
import sys
import time

import numpy as np


# --------------------------------------------------------------------------
# Progress logging
# --------------------------------------------------------------------------

_last_log_time = None


def log(message, indent=0):
    """
    Prints a timestamped, flushed progress message. flush=True matters
    here specifically because these scripts are usually invoked as
    subprocesses (build_feedpak.py -> process_vocals.py / process_gp_
    alignment.py) piping stdout back to a terminal — without an explicit
    flush, messages can sit buffered and appear to "hang" even though
    work is progressing, especially on Windows.
    """
    global _last_log_time
    now = time.time()
    prefix = "  " * indent + "[*]"
    print(f"{prefix} {message}", flush=True)
    _last_log_time = now


def log_step(step_num, total_steps, message):
    print(f"[{step_num}/{total_steps}] {message}", flush=True)


class timed_step:
    """
    Context manager: logs a start message, then an elapsed-time completion
    message. Use around any block that might take a while (model loads,
    DTW, audio decode) so long silent stretches are visible instead of
    looking stalled.

        with timed_step("Loading WhisperX model"):
            model = whisperx.load_model(...)
    """

    def __init__(self, message, indent=0):
        self.message = message
        self.indent = indent

    def __enter__(self):
        log(f"{self.message}...", indent=self.indent)
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self._start
        if exc_type is None:
            log(f"{self.message} — done ({elapsed:.1f}s)", indent=self.indent)
        else:
            log(f"{self.message} — failed after {elapsed:.1f}s", indent=self.indent)
        return False


# --------------------------------------------------------------------------
# Time / pitch conversion
# --------------------------------------------------------------------------

def tick_to_seconds(tick, tempo_events, ticks_per_quarter=960):
    """
    Convert an absolute tick position to seconds, given a sorted list of
    (tick, bpm) tempo-change events (the first event should be at tick 0
    representing the initial tempo). Segment-accumulation, not a single
    global-BPM multiply, so this is correct across mid-song tempo changes.

    This is deliberately a pure function over plain (tick, bpm) tuples,
    not guitarpro objects, so it's testable without the guitarpro lib and
    reusable for both GP-tick and (if ever needed) MIDI-tick sources.
    """
    if not tempo_events:
        return tick / ticks_per_quarter * 0.5  # 120bpm fallback

    accumulated = 0.0
    last_tick = tempo_events[0][0]
    active_bpm = tempo_events[0][1]

    for ev_tick, bpm in tempo_events[1:]:
        if tick <= ev_tick:
            break
        delta_ticks = ev_tick - last_tick
        accumulated += (delta_ticks / ticks_per_quarter) * (60.0 / active_bpm)
        last_tick = ev_tick
        active_bpm = bpm

    remaining_ticks = tick - last_tick
    accumulated += (remaining_ticks / ticks_per_quarter) * (60.0 / active_bpm)
    return accumulated


def hz_to_midi(hz):
    """Continuous MIDI note number (not rounded) from a frequency in Hz."""
    if hz is None or hz <= 0:
        return None
    return 69.0 + 12.0 * math.log2(hz / 440.0)


def remap_string_index(gp_string_number, string_count):
    """
    pyguitarpro's Note.string is 1-based, 1 = highest-pitched string.
    Feedpak's `s` is 0-based, 0 = lowest-pitched string.
    """
    return string_count - gp_string_number


# --------------------------------------------------------------------------
# Drum piece mapping (GM percussion note -> feedpak drum-tab piece label)
# --------------------------------------------------------------------------

GM_DRUM_PIECE_MAP = {
    35: "kick", 36: "kick",
    38: "snare", 40: "snare", 37: "snare",
    42: "hh_closed", 44: "hh_closed", 46: "hh_closed",
    41: "tom_floor", 43: "tom_floor",             # floor toms
    45: "tom_mid", 47: "tom_mid",                 # low/low-mid toms
    48: "tom_hi", 50: "tom_hi",                   # high-mid/high toms
    49: "crash_r", 57: "crash_r",
    52: "crash_r",   # china -> crash (closest in character: loud, trashy accent)
    55: "crash_r",   # splash -> crash
    51: "ride", 59: "ride", 53: "ride",           # ride, ride 2, ride bell -> ride
}

# Representative note for each reduced-kit piece, used only as a numeric
# fallback for percussion notes not in GM_DRUM_PIECE_MAP above (auxiliary
# percussion, cowbell, tambourine, vendor-specific extras, etc.) so that
# gm_drum_to_piece() never drops a note — everything resolves to the
# closest of these eight pieces.
_REDUCED_KIT_REFERENCE_NOTE = {
    "kick": 36, "snare": 38, "hh_closed": 42,
    "tom_floor": 42, "tom_mid": 46, "tom_hi": 49,
    "crash_r": 49, "ride": 51,
}

# Target kit: 1 kick + 4 pads (snare, tom_hi, tom_mid, tom_floor) + 3
# cymbals (hh_closed, crash_r, ride) — a Rock Band 3 Pro Drums-style
# reduced kit. Every drum note in the source, regardless of its original
# GM number, gets translated (exact table match, or nearest-neighbor
# fallback) to one of these eight. Order and ids/names match a working
# reference feedpak's drum_tab.json `kit` array exactly.
REDUCED_DRUM_KIT = [
    {"id": "kick", "name": "Kick"},
    {"id": "snare", "name": "Snare"},
    {"id": "hh_closed", "name": "Hi-hat (closed)"},
    {"id": "tom_hi", "name": "High Tom"},
    {"id": "ride", "name": "Ride"},
    {"id": "tom_mid", "name": "Mid Tom"},
    {"id": "crash_r", "name": "Crash (right)"},
    {"id": "tom_floor", "name": "Floor Tom"},
]


def gm_drum_to_piece(gm_note):
    """
    Maps any incoming GM percussion note to one of the reduced kit's
    eight pieces (kick, snare, hh_closed, tom_hi, tom_mid, tom_floor,
    crash_r, ride). Known GM percussion notes use the semantic table
    above (china/splash -> crash_r, ride bell -> ride, etc.); anything
    else falls back to whichever reference note is numerically closest,
    so no drum hit is ever silently dropped for having an unrecognized
    note number.
    """
    piece = GM_DRUM_PIECE_MAP.get(gm_note)
    if piece is not None:
        return piece
    return min(_REDUCED_KIT_REFERENCE_NOTE.items(),
               key=lambda kv: abs(kv[1] - gm_note))[0]


# --------------------------------------------------------------------------
# Standard tuning references (absolute MIDI pitch, low string first),
# used to turn a track's absolute string pitches into the relative-offset
# `tuning` array the arrangement/manifest schemas expect.
# --------------------------------------------------------------------------

STANDARD_GUITAR_TUNING_MIDI = [40, 45, 50, 55, 59, 64]   # E2 A2 D3 G3 B3 E4
STANDARD_BASS_TUNING_MIDI = [28, 33, 38, 43]              # E1 A1 D2 G2


def tuning_offsets_from_absolute(absolute_midi_low_to_high, is_bass):
    """
    absolute_midi_low_to_high: list of absolute MIDI pitches, index 0 =
    lowest-pitched string (i.e. already remapped to feedpak's convention).
    Returns per-string integer offsets from standard tuning. If the string
    count doesn't match the standard reference (e.g. 7-string guitar,
    5-string bass), pads/truncates the reference by extending downward
    (best-effort; flagged in the returned `nonstandard_count` bool).
    """
    reference = STANDARD_BASS_TUNING_MIDI if is_bass else STANDARD_GUITAR_TUNING_MIDI
    n = len(absolute_midi_low_to_high)
    nonstandard_count = n != len(reference)
    if n <= len(reference):
        ref = reference[:n]
    else:
        # extend downward in perfect fourths (the standard interval used
        # elsewhere in both tunings) for any extra low strings
        ref = list(reference)
        while len(ref) < n:
            ref.insert(0, ref[0] - 5)
        ref = ref[-n:] if len(ref) > n else ref
    offsets = [absolute_midi_low_to_high[i] - ref[i] for i in range(n)]
    return offsets, nonstandard_count


# --------------------------------------------------------------------------
# Key signature naming (GP's KeySignature enum is (accidentals, minorFlag))
# --------------------------------------------------------------------------

_MAJOR_KEY_BY_ACCIDENTALS = {
    -7: "Cb", -6: "Gb", -5: "Db", -4: "Ab", -3: "Eb", -2: "Bb", -1: "F",
    0: "C", 1: "G", 2: "D", 3: "A", 4: "E", 5: "B", 6: "F#", 7: "C#",
}
_MINOR_KEY_BY_ACCIDENTALS = {
    -7: "Ab", -6: "Eb", -5: "Bb", -4: "F", -3: "C", -2: "G", -1: "D",
    0: "A", 1: "E", 2: "B", 3: "F#", 4: "C#", 5: "G#", 6: "D#", 7: "A#",
}


def key_signature_to_name(accidentals, is_minor):
    """
    accidentals: -7..7 (circle-of-fifths position, matches
    notation.schema.json's `ks` field and GP's KeySignature.value[0]).
    is_minor: bool, GP's KeySignature.value[1] (0/1).
    """
    table = _MINOR_KEY_BY_ACCIDENTALS if is_minor else _MAJOR_KEY_BY_ACCIDENTALS
    key = table.get(accidentals)
    if key is None:
        return None
    return f"{key}m" if is_minor else key


# --------------------------------------------------------------------------
# Monotonic-path dedup for DTW warp maps
# --------------------------------------------------------------------------

def dedupe_monotonic_path(xs, ys):
    """
    Given parallel arrays from a DTW warping path (monotonic but not
    necessarily strictly increasing in x), group by x and take the median
    y, returning strictly-increasing-x, sorted parallel lists suitable for
    scipy.interpolate.interp1d.
    """
    if len(xs) == 0:
        return [], []
    buckets = {}
    for x, y in zip(xs, ys):
        buckets.setdefault(x, []).append(y)
    sorted_x = sorted(buckets.keys())
    out_x, out_y = [], []
    for x in sorted_x:
        vals = sorted(buckets[x])
        n = len(vals)
        median = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
        out_x.append(x)
        out_y.append(median)
    return out_x, out_y


# --------------------------------------------------------------------------
# JSON export helpers
# --------------------------------------------------------------------------

def file_sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data, indent=2):
    path = os.fspath(path)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=indent, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def omit_if_negligible(d, key, value, threshold=0.05):
    """Set d[key] = value only if value is present and >= threshold."""
    if value is not None and value >= threshold:
        d[key] = round(value, 4)


def drop_empty_lists(d, keys):
    """
    Remove any of `keys` from dict `d` whose value is an empty list —
    several feedpak schemas (arrangement.tempos, arrangement.phrases)
    treat present-but-empty as non-conformant; omit the key entirely
    instead.
    """
    for k in keys:
        if k in d and isinstance(d[k], list) and len(d[k]) == 0:
            del d[k]
    return d


# --------------------------------------------------------------------------
# POSIX relative path validation (manifest.schema.json's `relpath`)
# --------------------------------------------------------------------------

_RELPATH_RE = re.compile(r"^(?!/)(?!.*//)(?!.*:)(?!.*\\)(?!.*(^|/)\.\.(/|$)).+$")


def is_valid_relpath(path):
    return bool(_RELPATH_RE.match(path))


def to_posix_relpath(*parts):
    path = "/".join(p.strip("/\\") for p in parts if p)
    path = path.replace("\\", "/")
    if not is_valid_relpath(path):
        raise ValueError(f"Generated path is not a valid feedpak relpath: {path!r}")
    return path


# --------------------------------------------------------------------------
# Count-in padding: detect whether a stem starts "cold" (no lead-in
# silence) and, if so, prepend one so audio and chart data can both carry
# a real 4-beat count-in instead of starting exactly on beat 1.
# --------------------------------------------------------------------------

def detect_leading_silence_seconds(audio_path, threshold_amplitude=0.02,
                                    max_scan_seconds=10.0):
    """
    Returns how many seconds of near-silence precede the first transient
    in `audio_path`, capped at max_scan_seconds. threshold_amplitude is
    on a 0..1 float-sample scale (0.02 ~= -34dBFS) — deliberately loose,
    since a "silent" count-in bar can still carry faint bleed/noise floor
    and we only care whether there's a real gap before the song starts.
    """
    import soundfile as sf

    with sf.SoundFile(audio_path) as f:
        sr = f.samplerate
        frames_to_scan = min(f.frames, int(max_scan_seconds * sr))
        data = f.read(frames_to_scan, dtype="float32", always_2d=True)

    amplitude = np.abs(data).max(axis=1)  # mono-mixdown peak per sample
    above = np.where(amplitude > threshold_amplitude)[0]
    if len(above) == 0:
        return max_scan_seconds  # entire scanned window is silent
    return above[0] / sr


def pad_audio_with_silence(src_path, dst_path, pad_seconds):
    """
    Writes a copy of src_path to dst_path with pad_seconds of silence
    prepended, preserving sample rate and channel count.

    dst_path should use a .wav extension regardless of src_path's
    format — libsndfile (which soundfile wraps) has a known crash
    writing large OGG Vorbis files (confirmed: a ~3.5 minute stereo
    44.1kHz file segfaults on write, while the identical audio writes
    fine as WAV). Re-encoding to OGG would also cost a lossy
    decode/re-encode generation loss on top of that risk, so padded
    stems are written as WAV — lossless, and avoids the crash entirely.
    """
    import soundfile as sf

    data, sr = sf.read(src_path, dtype="float32", always_2d=True)
    pad_samples = int(round(pad_seconds * sr))
    silence = np.zeros((pad_samples, data.shape[1]), dtype="float32")
    padded = np.concatenate([silence, data], axis=0)
    sf.write(dst_path, padded, sr, format="WAV")















