"""
process_vocals.py

Script 1 of the feedpak pipeline. Runs WhisperX (transcription + word-level
alignment + speaker diarization) and CREPE (continuous pitch tracking) over
a vocal stem, aggregates pitch per word, and exports a diarization-intact
intermediate structure keyed by speaker_id.

The ML-dependent pieces (WhisperX, torchcrepe) are isolated behind small
wrapper functions with guarded imports, so the pure aggregation logic below
can be unit-tested without those (heavy, GPU-oriented) packages installed.

Usage:
    python process_vocals.py vocals.ogg --out intermediate_vocals.json
"""
import argparse

import feedpak_common as fc

try:
    import whisperx
except ImportError:
    whisperx = None

try:
    import torchcrepe
    import torch
except ImportError:
    torchcrepe = None
    torch = None


CREPE_CONFIDENCE_THRESHOLD = 0.5
CREPE_HOP_SECONDS = 0.010  # 10ms hop, per architecture doc


# --------------------------------------------------------------------------
# Pure aggregation logic (testable without whisperx/torchcrepe installed)
# --------------------------------------------------------------------------

def aggregate_pitch_for_window(start, end, contour_times, contour_hz, contour_confidence,
                                confidence_threshold=CREPE_CONFIDENCE_THRESHOLD):
    """
    Given a word's [start, end) time window and full-song CREPE contour
    arrays, returns the median Hz across confident frames in that window,
    or None if no frame in the window clears the confidence threshold.
    """
    hz_in_window = [
        hz for t, hz, conf in zip(contour_times, contour_hz, contour_confidence)
        if start <= t < end and conf >= confidence_threshold and hz > 0
    ]
    if not hz_in_window:
        return None
    hz_in_window.sort()
    n = len(hz_in_window)
    return hz_in_window[n // 2] if n % 2 else (hz_in_window[n // 2 - 1] + hz_in_window[n // 2]) / 2.0


def build_speaker_structure(diarized_words, contour_times, contour_hz, contour_confidence,
                             confidence_threshold=CREPE_CONFIDENCE_THRESHOLD):
    """
    diarized_words: list of {"speaker": str, "start": float, "end": float, "word": str}
    Returns: {"speakers": {speaker_id: {"words": [...], "contour": [...]}}}

    Diarization is kept intact per speaker_id all the way through export —
    the feedpak lyrics/vocal-pitch schemas have no per-word speaker field,
    so the split into per-speaker files has to happen upstream (in Script
    3, from this structure), not be reconstructed later from a flat list.
    """
    speakers = {}

    for w in diarized_words:
        spk = w["speaker"]
        speakers.setdefault(spk, {"words": [], "contour_indices": set()})

        median_hz = aggregate_pitch_for_window(
            w["start"], w["end"], contour_times, contour_hz, contour_confidence,
            confidence_threshold=confidence_threshold,
        )
        entry = {
            "t": round(w["start"], 4),
            "d": round(w["end"] - w["start"], 4),
            "w": w["word"],
        }
        if median_hz is not None:
            midi = fc.hz_to_midi(median_hz)
            entry["midi"] = round(midi)
        speakers[spk]["words"].append(entry)

        for i, t in enumerate(contour_times):
            if w["start"] <= t < w["end"]:
                speakers[spk]["contour_indices"].add(i)

    result = {"speakers": {}}
    for spk, data in speakers.items():
        contour = [
            {"t": round(contour_times[i], 4), "hz": round(contour_hz[i], 3)}
            for i in sorted(data["contour_indices"])
            if contour_confidence[i] >= confidence_threshold and contour_hz[i] > 0
        ]
        result["speakers"][spk] = {
            "words": sorted(data["words"], key=lambda e: e["t"]),
            "contour": contour,
        }
    return result


# --------------------------------------------------------------------------
# ML wrappers (require whisperx / torchcrepe at runtime)
# --------------------------------------------------------------------------

def run_whisperx(audio_path, device="cuda", batch_size=16, compute_type="float16",
                  hf_token=None):
    """
    Returns a flat list of {"speaker", "start", "end", "word"} dicts,
    combining WhisperX transcription, forced word alignment, and
    pyannote.audio speaker diarization.
    """
    if whisperx is None:
        raise RuntimeError("whisperx is required: pip install whisperx")

    with fc.timed_step("Loading WhisperX model (large-v2)", indent=1):
        model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    with fc.timed_step("Loading and decoding audio", indent=1):
        audio = whisperx.load_audio(audio_path)

    with fc.timed_step("Transcribing", indent=1):
        result = model.transcribe(audio, batch_size=batch_size)
    fc.log(f"Detected language: {result.get('language', '?')}", indent=1)

    with fc.timed_step("Loading forced-alignment model", indent=1):
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
    with fc.timed_step("Running forced word alignment", indent=1):
        result = whisperx.align(result["segments"], align_model, metadata, audio, device)

    with fc.timed_step("Loading speaker diarization model", indent=1):
        diarize_model = whisperx.diarize.DiarizationPipeline(
            token=hf_token, device=device
        )
    with fc.timed_step("Running speaker diarization", indent=1):
        diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    words = []
    skipped = 0
    for segment in result["segments"]:
        for w in segment.get("words", []):
            if "start" not in w or "end" not in w:
                skipped += 1
                continue  # whisperx leaves timing off some low-confidence words
            words.append({
                "speaker": w.get("speaker", "SPEAKER_00"),
                "start": w["start"],
                "end": w["end"],
                "word": w["word"].strip(),
            })
    speakers_found = sorted({w["speaker"] for w in words})
    fc.log(f"Got {len(words)} timed words across {len(speakers_found)} speaker(s) "
           f"{speakers_found} ({skipped} words skipped, no timing)", indent=1)
    return words


def run_crepe(audio_path, device="cuda", hop_seconds=CREPE_HOP_SECONDS):
    """
    Returns (times, hz, confidence) parallel arrays from torchcrepe.
    """
    if torchcrepe is None:
        raise RuntimeError("torchcrepe is required: pip install torchcrepe")

    import torchaudio

    with fc.timed_step("Loading audio for CREPE", indent=1):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

    hop_length = int(hop_seconds * sr)
    duration_s = audio.shape[-1] / sr
    fc.log(f"Running CREPE pitch tracking ({duration_s:.1f}s of audio, "
           f"{hop_seconds*1000:.0f}ms hop)...", indent=1)
    with fc.timed_step("CREPE inference", indent=1):
        pitch, periodicity = torchcrepe.predict(
            audio, sr, hop_length,
            fmin=50.0, fmax=1100.0, model="full",
            batch_size=2048, device=device, return_periodicity=True,
        )
    pitch = pitch.squeeze(0)
    periodicity = periodicity.squeeze(0)
    times = [i * hop_seconds for i in range(pitch.shape[0])]
    confident = sum(1 for c in periodicity.tolist() if c >= CREPE_CONFIDENCE_THRESHOLD)
    fc.log(f"{len(times)} pitch frames, {confident} above confidence threshold "
           f"({CREPE_CONFIDENCE_THRESHOLD})", indent=1)
    return times, pitch.tolist(), periodicity.tolist()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def process(vocals_path, device="cuda", hf_token=None):
    fc.log(f"Processing vocal stem: {vocals_path}")
    fc.log_step(1, 3, "WhisperX (transcribe + align + diarize)")
    words = run_whisperx(vocals_path, device=device, hf_token=hf_token)

    fc.log_step(2, 3, "CREPE pitch tracking")
    times, hz, confidence = run_crepe(vocals_path, device=device)

    fc.log_step(3, 3, "Aggregating pitch per word, splitting by speaker")
    result = build_speaker_structure(words, times, hz, confidence)
    for speaker_id, data in result["speakers"].items():
        with_pitch = sum(1 for w in data["words"] if "midi" in w)
        fc.log(f"{speaker_id}: {len(data['words'])} words "
               f"({with_pitch} with a resolved pitch), "
               f"{len(data['contour'])} contour samples", indent=1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract diarized lyrics + pitch from a vocal stem.")
    parser.add_argument("vocals_audio", help="Path to the isolated vocal stem (e.g. vocals.ogg)")
    parser.add_argument("--out", default="intermediate_vocals.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote diarization models")
    args = parser.parse_args()

    result = process(args.vocals_audio, device=args.device, hf_token=args.hf_token)
    fc.write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
