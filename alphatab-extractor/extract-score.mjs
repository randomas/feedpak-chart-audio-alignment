import * as fs from "node:fs";
import * as path from "node:path";
import * as alphaTab from "@coderline/alphatab";

const SCHEMA_VERSION = 4;
const PPQ = 960;
const args = process.argv.slice(2);
const positional = args.filter((x) => !x.startsWith("--"));
const includeCursorDetail = args.includes("--cursor-detail");
const beatTextAsLyrics = args.includes("--beat-text-lyrics");

if (positional.length < 1) {
    console.error("Usage: node extract-score.mjs <input.gp> [output.json] [--beat-text-lyrics] [--cursor-detail]");
    process.exit(2);
}

const inputPath = path.resolve(positional[0]);
const outputPath = path.resolve(positional[1] ?? "alphatab-score.json");
const arr = (value) => { if (!value) return []; if (Array.isArray(value)) return value; try { return Array.from(value); } catch { return []; } };
const num = (value, fallback = null) => Number.isFinite(value) ? Number(value) : fallback;
const bool = (value) => Boolean(value);
const simple = (value) => value === undefined || value === null ? null : String(value);
const enumName = (obj, value) => value === undefined || value === null ? null : obj && Object.prototype.hasOwnProperty.call(obj, value) ? String(obj[value]) : String(value);
const enumValue = (obj, value) => ({ value: num(value), name: enumName(obj, value) });

function exportAutomation(a) {
    return a ? { type: simple(a.type), value: num(a.value), ratio_position: num(a.ratioPosition), linear: bool(a.isLinear), text: a.text ?? null } : null;
}
function exportPoint(p) { return p ? { offset: num(p.offset), value: num(p.value) } : null; }
function exportLyrics(lyrics) {
    return arr(lyrics).map((x, lineIndex) => {
        const text = typeof x === "string" ? x : (x?.text ?? String(x ?? ""));
        return { line_index: lineIndex, text, is_empty: text.trim().length === 0, start: num(x?.start), next_text: x?.nextLyrics?.text ?? null, previous_text: x?.previousLyrics?.text ?? null };
    });
}
function nonEmptyLyrics(lyrics) { return exportLyrics(lyrics).filter((x) => !x.is_empty); }

function exportPercussionArticulation(articulation, index) {
    if (!articulation) return null;
    return {
        index,
        id: num(articulation.id),
        unique_id: articulation.uniqueId ?? null,
        element_type: articulation.elementType ?? null,
        output_midi_number: num(articulation.outputMidiNumber),
        staff_line: num(articulation.staffLine),
        notehead: {
            default: simple(articulation.noteHeadDefault),
            half: simple(articulation.noteHeadHalf),
            whole: simple(articulation.noteHeadWhole)
        },
        technique_symbol: simple(articulation.techniqueSymbol),
        technique_symbol_placement: simple(articulation.techniqueSymbolPlacement)
    };
}

function percussionResolver(track) {
    const list = arr(track?.percussionArticulations);
    const byId = new Map();
    const byIndex = new Map();
    list.forEach((articulation, index) => {
        byIndex.set(index, articulation);
        if (articulation?.id !== undefined && articulation?.id !== null) byId.set(Number(articulation.id), articulation);
    });
    return { list, byId, byIndex };
}

function resolvePercussion(track, reference) {
    if (reference === undefined || reference === null) return null;
    const resolver = percussionResolver(track);
    const key = Number(reference);
    const articulation = resolver.byIndex.get(key) ?? null;
    if (!articulation) return { articulation_reference: key, resolved: false, articulation_id: null, articulation_index: null, output_midi_number: null };
    return {
        articulation_reference: key,
        resolved: true,
        articulation_id: num(articulation.id),
        articulation_index: resolver.list.indexOf(articulation),
        output_midi_number: num(articulation.outputMidiNumber),
        element_type: articulation.elementType ?? null,
        staff_line: num(articulation.staffLine),
        unique_id: articulation.uniqueId ?? null
    };
}

function exportNote(note, beat, context) {
    const bendPoints = arr(note.bendPoints).map(exportPoint).filter(Boolean);
    const midi = Number.isFinite(note.realValue) ? Number(note.realValue) : Number.isFinite(note.displayValue) ? Number(note.displayValue) : Number.isFinite(note.value) ? Number(note.value) : null;
    return {
        id: num(note.id), source: { ...context },
        pitch: { midi, octave: num(note.octave), tone: num(note.tone), percussion_articulation: num(note.percussionArticulation) },
        tablature: { string: num(note.string), fret: num(note.fret), display_value: num(note.displayValue), display_value_without_bend: num(note.displayValueWithoutBend) },
        duration: { percent: num(note.durationPercent), beat_playback_duration: num(beat.playbackDuration), beat_display_duration: num(beat.displayDuration) },
        links: {
            tie_origin_id: num(note.tieOrigin?.id), tie_destination_id: num(note.tieDestination?.id), is_tie_origin: Boolean(note.tieDestination), is_tie_destination: Boolean(note.tieOrigin),
            hammer_pull_origin_id: num(note.hammerPullOrigin?.id), hammer_pull_destination_id: num(note.hammerPullDestination?.id), effect_slur_origin_id: num(note.effectSlurOrigin?.id), effect_slur_destination_id: num(note.effectSlurDestination?.id), bend_origin_id: num(note.bendOrigin?.id)
        },
        techniques: {
            accentuation: enumValue(alphaTab.model?.AccentuationType, note.accentuated), bend_type: enumValue(alphaTab.model?.BendType, note.bendType), bend_style: enumValue(alphaTab.model?.BendStyle, note.bendStyle), bend_points: bendPoints,
            slide_in_type: enumValue(alphaTab.model?.SlideInType, note.slideInType), slide_out_type: enumValue(alphaTab.model?.SlideOutType, note.slideOutType), harmonic_type: enumValue(alphaTab.model?.HarmonicType, note.harmonicType), harmonic_value: num(note.harmonicValue),
            vibrato: enumValue(alphaTab.model?.VibratoType, note.vibrato), trill_value: num(note.trillValue), trill_speed: enumValue(alphaTab.model?.Duration, note.trillSpeed), ornament: enumValue(alphaTab.model?.NoteOrnament, note.ornament), dynamics: enumValue(alphaTab.model?.DynamicValue, note.dynamics),
            left_hand_finger: enumValue(alphaTab.model?.Fingers, note.leftHandFinger), right_hand_finger: enumValue(alphaTab.model?.Fingers, note.rightHandFinger), dead: bool(note.isDead), ghost: bool(note.isGhost), let_ring: bool(note.isLetRing ?? beat.isLetRing), palm_mute: bool(note.isPalmMute ?? beat.isPalmMute), staccato: bool(note.isStaccato), tapping: bool(note.isTapping), slap: bool(note.isSlap), pop: bool(note.isPop), visible: note.isVisible !== false
        }
    };
}

function exportBeat(beat, context) {
    return {
        id: num(beat.id), index: num(beat.index, context.beat_index), source: { ...context },
        written_timing: { display_start: num(beat.displayStart), playback_start: num(beat.playbackStart), absolute_display_start: num(beat.absoluteDisplayStart), absolute_playback_start: num(beat.absolutePlaybackStart), display_duration: num(beat.displayDuration), playback_duration: num(beat.playbackDuration) },
        rhythm: { duration: enumValue(alphaTab.model?.Duration, beat.duration), dots: num(beat.dots, 0), tuplet_numerator: num(beat.tupletNumerator), tuplet_denominator: num(beat.tupletDenominator), grace_type: enumValue(alphaTab.model?.GraceType, beat.graceType), grace_index: num(beat.graceIndex), brush_duration: num(beat.brushDuration), brush_type: enumValue(alphaTab.model?.BrushType, beat.brushType), pick_stroke: enumValue(alphaTab.model?.PickStroke, beat.pickStroke), tremolo_speed: enumValue(alphaTab.model?.Duration, beat.tremoloSpeed) },
        state: { empty: bool(beat.isEmpty), rest: bool(beat.isRest), full_bar_rest: bool(beat.isFullBarRest), tremolo: bool(beat.isTremolo), let_ring: bool(beat.isLetRing), palm_mute: bool(beat.isPalmMute), whammy_bar: bool(beat.hasWhammyBar), rasgueado: bool(beat.hasRasgueado), legato_origin: bool(beat.isLegatoOrigin), legato_destination: bool(beat.isLegatoDestination), fade_in: bool(beat.fadeIn ?? beat.fade) },
        text: beat.text ?? null, lyrics: exportLyrics(beat.lyrics),
        chord: { has_chord: bool(beat.hasChord), chord_id: beat.chordId ?? null, barre_fret: num(beat.barreFret), barre_shape: simple(beat.barreShape) },
        dynamics: enumValue(alphaTab.model?.DynamicValue, beat.dynamics), crescendo: simple(beat.crescendo),
        whammy_bar_points: arr(beat.whammyBarPoints).map(exportPoint).filter(Boolean), automations: arr(beat.automations).map(exportAutomation).filter(Boolean),
        notes: arr(beat.notes).map((note, noteIndex) => exportNote(note, beat, { ...context, note_index: noteIndex }))
    };
}

function exportTrack(track, trackIndex) {
    const staves = arr(track.staves).map((staff, staffIndex) => ({
        index: staffIndex,
        properties: { is_percussion: bool(staff.isPercussion), capo: num(staff.capo, 0), tuning: arr(staff.tuning).map(Number), show_tablature: bool(staff.showTablature), show_standard_notation: bool(staff.showStandardNotation), standard_notation_line_count: num(staff.standardNotationLineCount), transposition_pitch: num(staff.transpositionPitch), display_transposition_pitch: num(staff.displayTranspositionPitch) },
        bars: arr(staff.bars).map((bar, barIndex) => ({
            id: num(bar.id), index: num(bar.index, barIndex), master_bar_index: num(bar.masterBar?.index), clef: enumValue(alphaTab.model?.Clef, bar.clef), clef_ottava: enumValue(alphaTab.model?.Ottavia, bar.clefOttava), simile_mark: enumValue(alphaTab.model?.SimileMark, bar.simileMark),
            voices: arr(bar.voices).map((voice, voiceIndex) => ({ index: voiceIndex, empty: bool(voice.isEmpty), beats: arr(voice.beats).map((beat, beatIndex) => exportBeat(beat, { track_index: trackIndex, staff_index: staffIndex, bar_index: barIndex, voice_index: voiceIndex, beat_index: beatIndex })) }))
        }))
    }));
    return {
        id: num(track.id), index: trackIndex, name: track.name ?? null, short_name: track.shortName ?? null,
        playback: { program: num(track.playbackInfo?.program), channel: num(track.playbackInfo?.primaryChannel), secondary_channel: num(track.playbackInfo?.secondaryChannel), volume: num(track.playbackInfo?.volume), balance: num(track.playbackInfo?.balance), port: num(track.playbackInfo?.port), solo: bool(track.playbackInfo?.isSolo), mute: bool(track.playbackInfo?.isMute) },
        percussion_articulations: arr(track.percussionArticulations).map(exportPercussionArticulation).filter(Boolean),
        staves
    };
}

function exportMasterBar(bar, index) {
    return {
        id: num(bar.id), index, display_number: index + 1,
        written_timing: { start: num(bar.start), duration: typeof bar.calculateDuration === "function" ? num(bar.calculateDuration()) : null, anacrusis: bool(bar.isAnacrusis), free_time: bool(bar.isFreeTime) },
        meter: { numerator: num(bar.timeSignatureNumerator), denominator: num(bar.timeSignatureDenominator), common_time: bool(bar.timeSignatureCommon), triplet_feel: simple(bar.tripletFeel) },
        key: { signature: num(bar.keySignature), type: simple(bar.keySignatureType) },
        navigation: { repeat_start: bool(bar.isRepeatStart), repeat_end: bool(bar.isRepeatEnd), repeat_count: num(bar.repeatCount, 0), alternate_endings: num(bar.alternateEndings, 0), directions: arr(bar.directions).map(simple) },
        section: bar.section ? { marker: bar.section.marker ?? null, text: bar.section.text ?? null } : null,
        tempo_automation: exportAutomation(bar.tempoAutomation), tempo_automations: arr(bar.tempoAutomations).map(exportAutomation).filter(Boolean), fermatas: arr(bar.fermatas).map((f) => ({ type: simple(f?.type), length: num(f?.length) }))
    };
}

function walkBeats(score) {
    const out = [];
    arr(score.tracks).forEach((track, trackIndex) => arr(track.staves).forEach((staff, staffIndex) => arr(staff.bars).forEach((bar, barIndex) => arr(bar.voices).forEach((voice, voiceIndex) => arr(voice.beats).forEach((beat, beatIndex) => out.push({ beat, trackIndex, staffIndex, barIndex, voiceIndex, beatIndex }))))));
    return out;
}

function buildPlayback(score, settings) {
    const output = { available: false, error: null, diagnostics: {}, master_bar_visits: [], beat_occurrences: [], playback_notes: [], tempo_events: [], sync_points: [] };
    if (includeCursorDetail) output.cursor_beat_visits = [];
    try {
        const MidiFile = alphaTab.midi?.MidiFile;
        const Handler = alphaTab.midi?.AlphaSynthMidiFileHandler;
        const Generator = alphaTab.midi?.MidiFileGenerator;
        if (!MidiFile || !Handler || !Generator) throw new Error("Required alphaTab MIDI classes are not exported.");
        const generator = new Generator(score, settings, new Handler(new MidiFile()));
        generator.generate();
        const lookup = generator.tickLookup;
        if (!lookup) throw new Error("MIDI generation completed without tickLookup.");
        const contexts = new Map();
        for (const x of walkBeats(score)) contexts.set(x.beat, { track_index: x.trackIndex, staff_index: x.staffIndex, source_bar_index: x.barIndex, voice_index: x.voiceIndex, beat_index: x.beatIndex, beat_id: num(x.beat.id) });
        const visits = arr(lookup.masterBars);
        const occurrenceCounts = new Map();
        const beatKeys = new Set();
        visits.forEach((visit, playbackIndex) => {
            const masterBar = visit.masterBar ?? visit.bar ?? null;
            const sourceIndex = num(masterBar?.index);
            const occurrence = (occurrenceCounts.get(sourceIndex) ?? 0) + 1;
            occurrenceCounts.set(sourceIndex, occurrence);
            output.master_bar_visits.push({ playback_index: playbackIndex, source_master_bar_index: sourceIndex, occurrence, start_tick: num(visit.start), end_tick: num(visit.end), duration_ticks: Number.isFinite(visit.start) && Number.isFinite(visit.end) ? Number(visit.end) - Number(visit.start) : num(visit.duration) });
            let beatLookup = visit.firstBeat ?? visit.firstBeatLookup ?? null;
            const seen = new Set();
            while (beatLookup && !seen.has(beatLookup)) {
                seen.add(beatLookup);
                for (const item of arr(beatLookup.highlightedBeats)) {
                    const beat = item.beat ?? null;
                    const context = contexts.get(beat) ?? { source_bar_index: sourceIndex, beat_id: num(beat?.id) };
                    const relativeStart = num(item.playbackStart, 0);
                    const absoluteStart = num(visit.start, 0) + relativeStart;
                    const key = [playbackIndex, context.track_index, context.staff_index, context.voice_index, context.beat_id, relativeStart].join(":");
                    if (!beatKeys.has(key)) {
                        beatKeys.add(key);
                        output.beat_occurrences.push({ playback_master_bar_index: playbackIndex, source_master_bar_index: sourceIndex, occurrence, ...context, relative_start_tick: relativeStart, absolute_start_tick: absoluteStart, written_absolute_playback_tick: num(beat?.absolutePlaybackStart), playback_duration_ticks: num(beat?.playbackDuration), display_duration_ticks: num(beat?.displayDuration), is_rest: bool(beat?.isRest), is_empty: bool(beat?.isEmpty), is_full_bar_rest: bool(beat?.isFullBarRest), lyric_fragments: nonEmptyLyrics(beat?.lyrics), note_ids: arr(beat?.notes).map((n) => num(n.id)).filter((x) => x !== null) });
                        for (const note of arr(beat?.notes)) {
                            const track = arr(score.tracks)[context.track_index];
                            const percussion = resolvePercussion(track, note.percussionArticulation);
                            output.playback_notes.push({
                                playback_master_bar_index: playbackIndex, source_master_bar_index: sourceIndex, occurrence,
                                track_index: context.track_index ?? null, staff_index: context.staff_index ?? null, voice_index: context.voice_index ?? null, beat_index: context.beat_index ?? null,
                                beat_id: context.beat_id ?? null, note_id: num(note.id), absolute_start_tick: absoluteStart, duration_ticks: num(beat?.playbackDuration),
                                pitch_midi: num(note.realValue ?? note.displayValue ?? note.value), string: num(note.string), fret: num(note.fret), percussion
                            });
                        }
                    }
                    if (includeCursorDetail) output.cursor_beat_visits.push({ playback_master_bar_index: playbackIndex, source_master_bar_index: sourceIndex, occurrence, ...context, lookup_start_tick: num(beatLookup.start), lookup_end_tick: num(beatLookup.end), beat_playback_start_tick: relativeStart });
                }
                beatLookup = beatLookup.nextBeat;
            }
        });
        for (const visit of visits) for (const change of arr(visit.tempoChanges)) output.tempo_events.push({ tick: num(visit.start, 0) + num(change.tick ?? change.offset, 0), tempo: num(change.tempo ?? change.value) });
        output.tempo_events = output.tempo_events.filter((x) => x.tempo !== null).sort((a, b) => a.tick - b.tick).filter((x, i, all) => i === 0 || x.tick !== all[i - 1].tick || x.tempo !== all[i - 1].tempo);
        output.sync_points = arr(generator.syncPoints).map((p) => ({ bar_index: num(p.barIndex), bar_position: num(p.barPosition), millisecond_offset: num(p.millisecondOffset), absolute_time: num(p.absoluteTime) }));
        output.available = true;
        output.summary = { playback_master_bar_count: output.master_bar_visits.length, playback_beat_occurrence_count: output.beat_occurrences.length, playback_note_count: output.playback_notes.length, cursor_beat_visit_count: includeCursorDetail ? output.cursor_beat_visits.length : null, tempo_event_count: output.tempo_events.length, sync_point_count: output.sync_points.length };
    } catch (error) { output.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error); }
    return output;
}

function buildLyricReport(tracks, playback) {
    const firstOccurrence = new Map();
    for (const o of playback.beat_occurrences ?? []) if (!firstOccurrence.has(o.beat_id) || o.absolute_start_tick < firstOccurrence.get(o.beat_id).absolute_start_tick) firstOccurrence.set(o.beat_id, o);
    let totalSlots = 0, totalFragments = 0, totalBeats = 0;
    const allLines = new Set();
    const trackReports = tracks.map((track) => {
        let slots = 0, lyricBeats = 0;
        const fragments = [], beatTexts = [], lines = new Set();
        for (const staff of track.staves) for (const bar of staff.bars) for (const voice of bar.voices) for (const beat of voice.beats) {
            slots += beat.lyrics.length;
            const meaningful = beat.lyrics.filter((x) => !x.is_empty);
            if (meaningful.length > 0) lyricBeats += 1;
            const occurrence = firstOccurrence.get(beat.id);
            for (const lyric of meaningful) {
                lines.add(lyric.line_index); allLines.add(lyric.line_index);
                fragments.push({ track_index: track.index, track_name: track.name, staff_index: staff.index, bar_index: bar.index, voice_index: voice.index, beat_index: beat.index, beat_id: beat.id, line_index: lyric.line_index, text: lyric.text, lyric_start: lyric.start, written_tick: beat.written_timing.absolute_playback_start, first_playback_tick: occurrence?.absolute_start_tick ?? null, playback_duration_ticks: beat.written_timing.playback_duration, note_ids: beat.notes.map((n) => n.id), note_pitches_midi: beat.notes.map((n) => n.pitch.midi).filter((x) => x !== null) });
            }
            if (beat.text && beat.text.trim()) beatTexts.push({ staff_index: staff.index, bar_index: bar.index, voice_index: voice.index, beat_index: beat.index, beat_id: beat.id, text: beat.text, written_tick: beat.written_timing.absolute_playback_start, first_playback_tick: occurrence?.absolute_start_tick ?? null });
        }
        totalSlots += slots; totalFragments += fragments.length; totalBeats += lyricBeats;
        return { track_index: track.index, track_name: track.name, lyric_slot_count: slots, non_empty_lyric_fragment_count: fragments.length, lyric_bearing_beat_count: lyricBeats, active_lyric_line_indices: [...lines].sort((a, b) => a - b), beat_text_count: beatTexts.length, lyric_fragments: fragments, beat_texts: beatTexts };
    });
    return { lyric_slot_count: totalSlots, non_empty_lyric_fragment_count: totalFragments, lyric_bearing_beat_count: totalBeats, active_lyric_line_indices: [...allLines].sort((a, b) => a - b), total_beat_texts: trackReports.reduce((s, t) => s + t.beat_text_count, 0), tracks: trackReports };
}

function buildExpressionReport(tracks) {
    const counts = {}; let writtenNoteCount = 0;
    const add = (name, condition) => { if (condition) counts[name] = (counts[name] ?? 0) + 1; };
    for (const track of tracks) for (const staff of track.staves) for (const bar of staff.bars) for (const voice of bar.voices) for (const beat of voice.beats) {
        add("tremolo_beats", beat.state.tremolo); add("whammy_beats", beat.state.whammy_bar || beat.whammy_bar_points.length > 0); add("rasgueado_beats", beat.state.rasgueado); add("grace_beats", (beat.rhythm.grace_type?.value ?? 0) > 0); add("brush_or_strum_beats", (beat.rhythm.brush_duration ?? 0) > 0); add("pick_stroke_beats", (beat.rhythm.pick_stroke?.value ?? 0) > 0);
        for (const note of beat.notes) { writtenNoteCount += 1; const t = note.techniques; add("bend_notes", t.bend_points.length > 0 || (t.bend_type?.value ?? 0) > 0); add("slide_notes", (t.slide_in_type?.value ?? 0) > 0 || (t.slide_out_type?.value ?? 0) > 0); add("harmonic_notes", (t.harmonic_type?.value ?? 0) > 0); add("vibrato_notes", (t.vibrato?.value ?? 0) > 0); add("trill_notes", t.trill_value !== null && t.trill_value >= 0); add("hammer_pull_linked_notes", note.links.hammer_pull_origin_id !== null || note.links.hammer_pull_destination_id !== null); add("tie_notes", note.links.is_tie_origin || note.links.is_tie_destination); add("dead_notes", t.dead); add("ghost_notes", t.ghost); add("let_ring_notes", t.let_ring || beat.state.let_ring); add("palm_mute_notes", t.palm_mute || beat.state.palm_mute); add("staccato_notes", t.staccato); add("tapping_notes", t.tapping); add("slap_notes", t.slap); add("pop_notes", t.pop); add("fingered_notes", (t.left_hand_finger?.value ?? 0) > 0 || (t.right_hand_finger?.value ?? 0) > 0); }
    }
    return { written_note_count: writtenNoteCount, counts };
}

async function main() {
    if (!fs.existsSync(inputPath)) throw new Error(`Input file not found: ${inputPath}`);
    const settings = new alphaTab.Settings();
    if (settings.importer) settings.importer.beatTextAsLyrics = beatTextAsLyrics;
    const bytes = await fs.promises.readFile(inputPath);
    const score = alphaTab.importer.ScoreLoader.loadScoreFromBytes(new Uint8Array(bytes), settings);
    const masterBars = arr(score.masterBars).map(exportMasterBar);
    const tracks = arr(score.tracks).map(exportTrack);
    const playback = buildPlayback(score, settings);
    const lyricReport = buildLyricReport(tracks, playback);
    const expressionReport = buildExpressionReport(tracks);
    const output = {
        schema_version: SCHEMA_VERSION,
        source: { file: inputPath, parser: "alphatab", parser_version: "1.8.3", ppq: PPQ, beat_text_as_lyrics: beatTextAsLyrics, cursor_detail_included: includeCursorDetail },
        metadata: { title: score.title ?? null, subtitle: score.subTitle ?? null, artist: score.artist ?? null, album: score.album ?? null, words: score.words ?? null, music: score.music ?? null, copyright: score.copyright ?? null, notices: score.notices ?? null, instructions: score.instructions ?? null, tempo: num(score.tempo), tempo_label: score.tempoLabel ?? null },
        summary: { track_count: tracks.length, master_bar_count: masterBars.length, playback_master_bar_count: playback.summary?.playback_master_bar_count ?? null, playback_beat_occurrence_count: playback.summary?.playback_beat_occurrence_count ?? null, playback_note_count: playback.summary?.playback_note_count ?? null, has_pickup: masterBars.some((b) => b.written_timing.anacrusis), has_repeats: masterBars.some((b) => b.navigation.repeat_start || b.navigation.repeat_end || b.navigation.repeat_count > 0), has_alternate_endings: masterBars.some((b) => b.navigation.alternate_endings > 0), non_empty_lyric_fragment_count: lyricReport.non_empty_lyric_fragment_count, lyric_bearing_beat_count: lyricReport.lyric_bearing_beat_count },
        master_bars: masterBars, playback, tracks, lyric_report: lyricReport, expression_report: expressionReport
    };
    await fs.promises.writeFile(outputPath, JSON.stringify(output, null, 2), "utf8");
    console.log(`Loaded: ${output.metadata.title ?? inputPath}`);
    console.log(`Tracks: ${tracks.length}`);
    console.log(`Written master bars: ${masterBars.length}`);
    if (playback.available) {
        console.log(`Playback master-bar visits: ${playback.summary.playback_master_bar_count}`);
        console.log(`Playback beat occurrences: ${playback.summary.playback_beat_occurrence_count}`);
        console.log(`Playback notes: ${playback.summary.playback_note_count}`);
        console.log(`Tempo events: ${playback.summary.tempo_event_count}`);
    } else console.warn(`Playback lookup unavailable: ${playback.error}`);
    console.log(`Non-empty lyric fragments: ${lyricReport.non_empty_lyric_fragment_count}`);
    console.log(`Lyric-bearing beats: ${lyricReport.lyric_bearing_beat_count}`);
    console.log(`Lyric slots: ${lyricReport.lyric_slot_count}`);
    console.log(`Beat texts: ${lyricReport.total_beat_texts}`);
    console.log(`Written notes: ${expressionReport.written_note_count}`);
    for (const track of tracks) console.log(`  ${track.index + 1}. ${track.name} (${track.staves.length} stave(s), ${track.percussion_articulations.length} percussion articulation(s))`);
    console.log(`Wrote: ${outputPath}`);
}

main().catch((error) => { console.error(error instanceof Error ? error.stack : String(error)); process.exit(1); });
