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

try:
    import pyphen
except ImportError:
    pyphen = None


CREPE_CONFIDENCE_THRESHOLD = 0.5
CREPE_HOP_SECONDS = 0.010  # 10ms hop, per architecture doc
VOCAL_ANALYSIS_VERSION = 3
MIN_VOCAL_NOTE_SECONDS = 0.150
MIN_PITCH_CHANGE_SEMITONES = 1.0
MAX_VOICED_GAP_SECONDS = 0.060
MIN_EMITTED_NOTE_SECONDS = 0.040

# WhisperX language codes -> pyphen dictionary names. Falls back to
# en_US for anything not listed (pyphen's dictionary set is smaller than
# Whisper's language list, and a wrong-language hyphenation dictionary
# fails safe anyway — see syllabify_word()'s fallback).
_PYPHEN_LANG_MAP = {
    "en": "en_US", "de": "de_DE", "fr": "fr_FR", "es": "es_ES", "it": "it_IT",
    "pt": "pt_PT", "nl": "nl_NL", "sv": "sv_SE", "ru": "ru_RU", "pl": "pl_PL",
    "da": "da_DK", "fi": "fi_FI", "nb": "nb_NO", "el": "el_GR",
}
_SYLLABLE_STRIP_CHARS = ".,!?;:\"'()[]"


# --------------------------------------------------------------------------
# Pure aggregation logic (testable without whisperx/torchcrepe installed)
# --------------------------------------------------------------------------

def get_syllable_dictionary(whisper_language_code):
    """
    Returns a pyphen.Pyphen dictionary for the detected transcription
    language, or None if pyphen isn't installed or the language has no
    dictionary — callers must treat None as "don't syllabify" (see
    syllabify_word()), not raise, since a missing hyphenation dictionary
    shouldn't take down the whole vocal pipeline.
    """
    if pyphen is None:
        return None
    lang = _PYPHEN_LANG_MAP.get(whisper_language_code, "en_US")
    try:
        return pyphen.Pyphen(lang=lang)
    except KeyError:
        try:
            return pyphen.Pyphen(lang="en_US")
        except KeyError:
            return None


def syllabify_word(word, dic):
    """
    Splits a transcribed word into syllables (e.g. "Montreux" ->
    ["Mon-", "treux"]), matching the standard vocal-chart convention of
    a trailing hyphen on every syllable but the last. Falls back to the
    whole word as a single "syllable" — no hyphen — whenever splitting
    isn't safe: no dictionary available, the word has no alphabetic
    core (numbers, pure punctuation), it's short enough that a split
    would be spurious, or the dictionary finds no break point.
    """
    if dic is None:
        return [word]

    leading = ""
    trailing = ""
    core = word
    while core and core[0] in _SYLLABLE_STRIP_CHARS:
        leading += core[0]
        core = core[1:]
    while core and core[-1] in _SYLLABLE_STRIP_CHARS:
        trailing = core[-1] + trailing
        core = core[:-1]

    if not core.isalpha() or len(core) <= 3:
        return [word]

    hyphenated = dic.inserted(core, hyphen="\x01")
    parts = [p for p in hyphenated.split("\x01") if p]
    if len(parts) <= 1:
        return [word]

    parts[0] = leading + parts[0]
    parts[-1] = parts[-1] + trailing
    return [p + "-" for p in parts[:-1]] + [parts[-1]]


def distribute_word_time_across_syllables(start, end, syllables):
    """
    Splits a word's [start, end) alignment window into one sub-window
    per syllable, proportional to each syllable's character count (we
    only have word-level timestamps from forced alignment, not
    phoneme-level, so this is a deliberate approximation — good enough
    to place syllable onsets in the right neighborhood, not a claim of
    phonetic precision). Returns [(syl_start, syl_end, syllable_text), ...].
    """
    core_lens = [max(1, len(s.rstrip("-"))) for s in syllables]
    total = sum(core_lens)
    duration = end - start
    spans = []
    t = start
    for syl, clen in zip(syllables, core_lens):
        d = duration * (clen / total)
        spans.append((t, t + d, syl))
        t += d
    return spans


def segment_contour_into_notes(times, hz, confidence,
                               conf_threshold=CREPE_CONFIDENCE_THRESHOLD,
                               min_note_seconds=MIN_VOCAL_NOTE_SECONDS,
                               min_pitch_change_semitones=MIN_PITCH_CHANGE_SEMITONES,
                               max_gap_seconds=MAX_VOICED_GAP_SECONDS):
    """Find stable, sequential pitch regions in one syllable.

    A pitch change must be at least one semitone and persist for at least
    ``min_note_seconds``. Brief excursions are treated as vibrato/noise.
    The returned regions are evidence only; ``partition_syllable_pitch``
    expands their boundaries to cover the complete pitched syllable.
    """
    frames=[]
    for t,h,c in zip(times,hz,confidence):
        if c >= conf_threshold and h > 0:
            midi=fc.hz_to_midi(h)
            if midi is not None:
                frames.append((float(t),float(midi),float(c)))
    if not frames:
        return []

    runs=[]; current=[]
    for frame in frames:
        if not current:
            current=[frame]; continue
        ref=sorted(x[1] for x in current)[len(current)//2]
        pitch_changed=abs(round(frame[1])-round(ref)) >= min_pitch_change_semitones
        gap=frame[0]-current[-1][0]
        if gap <= max_gap_seconds and not pitch_changed:
            current.append(frame)
        else:
            runs.append(current); current=[frame]
    if current: runs.append(current)

    def duration(run): return run[-1][0]-run[0][0]+CREPE_HOP_SECONDS
    def pitch(run):
        vals=sorted(x[1] for x in run); return int(round(vals[len(vals)//2]))

    # Remove short excursions. An A-short-B pattern joins the short run to
    # the closest neighbour, then adjacent equal-pitch runs are coalesced.
    changed=True
    while changed and len(runs)>1:
        changed=False
        for i,run in enumerate(runs):
            if duration(run) >= min_note_seconds: continue
            neighbours=[]
            if i>0: neighbours.append((abs(pitch(run)-pitch(runs[i-1])),i-1))
            if i+1<len(runs): neighbours.append((abs(pitch(run)-pitch(runs[i+1])),i+1))
            if not neighbours: continue
            _,j=min(neighbours,key=lambda x:(x[0],x[1]))
            runs[j]=sorted(runs[j]+run,key=lambda x:x[0]); runs.pop(i)
            runs.sort(key=lambda r:r[0][0]); changed=True; break

    stable=[]
    for run in runs:
        if duration(run) < min_note_seconds: continue
        stable.append({"start":run[0][0],"end":run[-1][0]+CREPE_HOP_SECONDS,
                       "midi":pitch(run),
                       "confidence":sum(x[2] for x in run)/len(run)})
    # Coalesce adjacent equal-pitch regions after short-run suppression.
    out=[]
    for seg in stable:
        if out and seg["midi"]==out[-1]["midi"] and seg["start"]<=out[-1]["end"]+max_gap_seconds:
            out[-1]["end"]=max(out[-1]["end"],seg["end"])
            out[-1]["confidence"]=max(out[-1]["confidence"],seg["confidence"])
        else: out.append(seg)
    return out


def partition_syllable_pitch(syllable, segments, fallback_midi, fallback_confidence):
    """Make a complete, gap-free, monophonic pitch partition for a syllable."""
    start=float(syllable["t"]); end=start+float(syllable["d"])
    if end <= start or fallback_midi is None: return []
    if not segments:
        return [{"t":round(start,4),"d":round(end-start,4),"midi":int(fallback_midi),
                 "confidence":round(float(fallback_confidence),4)}]
    segments=sorted(segments,key=lambda x:x["start"])
    # Boundaries are midpoints between adjacent evidence regions. The first
    # and last notes extend to the syllable edges, restoring renderer coverage.
    boundaries=[start]
    for a,b in zip(segments,segments[1:]):
        boundaries.append(max(start,min(end,(float(a["end"])+float(b["start"]))/2.0)))
    boundaries.append(end)
    notes=[]
    for i,seg in enumerate(segments):
        a=max(start,boundaries[i]); b=min(end,boundaries[i+1])
        if b-a < MIN_EMITTED_NOTE_SECONDS:
            continue
        notes.append({"t":round(a,4),"d":round(b-a,4),"midi":int(seg["midi"]),
                      "confidence":round(float(seg.get("confidence",fallback_confidence)),4)})
    if not notes:
        return [{"t":round(start,4),"d":round(end-start,4),"midi":int(fallback_midi),
                 "confidence":round(float(fallback_confidence),4)}]
    # Guarantee exact syllable coverage after any very-short partition removal.
    notes[0]["t"]=round(start,4)
    cursor=start
    for i,n in enumerate(notes):
        n["t"]=round(cursor,4)
        next_start=(float(notes[i+1]["t"]) if i+1<len(notes) else end)
        if i+1<len(notes):
            next_start=max(cursor+MIN_EMITTED_NOTE_SECONDS,float(notes[i+1]["t"]))
        n["d"]=round(max(MIN_EMITTED_NOTE_SECONDS,next_start-cursor),4)
        cursor=n["t"]+n["d"]
    notes[-1]["d"]=round(max(MIN_EMITTED_NOTE_SECONDS,end-notes[-1]["t"]),4)
    return notes

def expand_word(word_entry, syllable_dic):
    syllables=syllabify_word(word_entry["word"],syllable_dic)
    spans=distribute_word_time_across_syllables(word_entry["start"],word_entry["end"],syllables)
    return [{"t":round(a,4),"d":round(b-a,4),"w":w} for a,b,w in spans]


def representative_pitch(syllable,times,hz,confidence,threshold=CREPE_CONFIDENCE_THRESHOLD):
    start=syllable["t"]; end=start+syllable["d"]
    vals=sorted(fc.hz_to_midi(h) for t,h,c in zip(times,hz,confidence)
                if start<=t<end and c>=threshold and h>0)
    return int(round(vals[len(vals)//2])) if vals else None


def _resolve_monophonic_notes(notes, minimum_seconds=MIN_EMITTED_NOTE_SECONDS):
    """Merge same-pitch overlaps and trim different-pitch overlaps by confidence."""
    out=[]; suppressed=0
    for raw in sorted(notes,key=lambda n:(float(n["t"]),-float(n.get("confidence",0)))):
        n=dict(raw); n["t"]=float(n["t"]); n["d"]=float(n["d"])
        if n["d"]<=0: continue
        if not out: out.append(n); continue
        p=out[-1]; pe=p["t"]+p["d"]; ne=n["t"]+n["d"]
        if n["t"]>=pe-0.001:
            if n["t"]<pe: n["t"]=pe; n["d"]=ne-pe
            if n["d"]>=minimum_seconds: out.append(n)
            continue
        suppressed+=1
        if int(n["midi"])==int(p["midi"]):
            p["d"]=max(pe,ne)-p["t"]
            p["confidence"]=max(float(p.get("confidence",0)),float(n.get("confidence",0)))
        elif float(n.get("confidence",0))>float(p.get("confidence",0)):
            p["d"]=max(0,n["t"]-p["t"])
            if p["d"]<minimum_seconds: out.pop()
            out.append(n)
        else:
            n["t"]=pe; n["d"]=max(0,ne-pe)
            if n["d"]>=minimum_seconds: out.append(n)
    for n in out: n["t"]=round(n["t"],4); n["d"]=round(n["d"],4)
    return out,suppressed


def build_speaker_structure(diarized_words,contour_times,contour_hz,contour_confidence,
                             confidence_threshold=CREPE_CONFIDENCE_THRESHOLD,
                             whisper_language_code="en"):
    dic=get_syllable_dictionary(whisper_language_code); speakers={}
    for word in diarized_words:
        sp=word["speaker"]; speakers.setdefault(sp,{"words":[],"indices":set()})
        speakers[sp]["words"].extend(expand_word(word,dic))
        for i,t in enumerate(contour_times):
            if word["start"]<=t<word["end"]: speakers[sp]["indices"].add(i)
    result={"analysis_version":VOCAL_ANALYSIS_VERSION,"language":whisper_language_code,
            "pitch_settings":{"minimum_change_semitones":MIN_PITCH_CHANGE_SEMITONES,
                              "minimum_stable_seconds":MIN_VOCAL_NOTE_SECONDS,
                              "maximum_gap_seconds":MAX_VOICED_GAP_SECONDS},
            "speakers":{},"pitch_diagnostics":{}}
    for sp,data in speakers.items():
        words=sorted(data["words"],key=lambda x:x["t"]); candidates=[]
        pitched_syllables=multi_note_syllables=0
        for syl in words:
            start=float(syl["t"]); end=start+float(syl["d"])
            indices=[i for i,t in enumerate(contour_times) if start<=t<end]
            evidence=[(contour_times[i],contour_hz[i],contour_confidence[i]) for i in indices
                      if contour_confidence[i]>=confidence_threshold and contour_hz[i]>0]
            if not evidence: continue
            pitched_syllables+=1
            midis=sorted(fc.hz_to_midi(h) for _,h,_ in evidence)
            fallback=int(round(midis[len(midis)//2]))
            fallback_conf=sum(c for _,_,c in evidence)/len(evidence)
            segments=segment_contour_into_notes([x[0] for x in evidence],[x[1] for x in evidence],
                                                [x[2] for x in evidence],confidence_threshold)
            partition=partition_syllable_pitch(syl,segments,fallback,fallback_conf)
            if len(partition)>1: multi_note_syllables+=1
            candidates.extend(partition)
        notes,suppressed=_resolve_monophonic_notes(candidates)
        contour=[{"t":round(contour_times[i],4),"hz":round(contour_hz[i],3)}
                 for i in sorted(data["indices"])
                 if contour_confidence[i]>=confidence_threshold and contour_hz[i]>0]
        result["speakers"][sp]={"words":words,"pitch_notes":notes,"contour":contour}
        result["pitch_diagnostics"][sp]={"pitched_syllables":pitched_syllables,
             "multi_note_syllables":multi_note_syllables,"candidate_notes":len(candidates),
             "final_notes":len(notes),"overlaps_resolved":suppressed}
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
    detected_language = result["language"]
    fc.log(f"Detected language: {detected_language}", indent=1)

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
    return words, detected_language


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

def process(vocals_path, device="cuda", hf_token=None, batch_size=16, compute_type=None, language=None):
    fc.log(f"Processing vocal stem: {vocals_path}")
    fc.log_step(1, 3, "WhisperX (transcribe + align + diarize)")
    compute_type = compute_type or ("float16" if device == "cuda" else "int8")
    words, detected_language = run_whisperx(vocals_path, device=device, batch_size=batch_size,
                                             compute_type=compute_type, hf_token=hf_token)
    language = language or detected_language

    fc.log_step(2, 3, "CREPE pitch tracking")
    times, hz, confidence = run_crepe(vocals_path, device=device)

    fc.log_step(3, 3, "Splitting into syllables, detecting melisma, splitting by speaker")
    result = build_speaker_structure(words, times, hz, confidence, whisper_language_code=language)
    for speaker_id, data in result["speakers"].items():
        fc.log(f"{speaker_id}: {len(data['words'])} syllables, {len(data.get('pitch_notes',[]))} pitch notes, {len(data['contour'])} contour samples",indent=1)
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract diarized lyrics + pitch from a vocal stem.")
    parser.add_argument("vocals_audio", help="Path to the isolated vocal stem (e.g. vocals.ogg)")
    parser.add_argument("--out", default="intermediate_vocals.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--language", default=None)
    parser.add_argument("--hf-token", default=None, help="HuggingFace token for pyannote diarization models")
    args = parser.parse_args()

    result = process(args.vocals_audio, device=args.device, hf_token=args.hf_token,
                     batch_size=args.batch_size, compute_type=args.compute_type,
                     language=args.language)
    result["cache"] = {"audio_sha256": fc.file_sha256(args.vocals_audio)}
    fc.write_json(args.out, result)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()





















