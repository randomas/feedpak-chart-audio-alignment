from pathlib import Path
import pytest
import midi_section_fit as msf
import transcription_investigator as ti

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={
    'dtw-checkpoint','checkpoint-linear','checkpoint-dtw-diagnostic',
    'checkpoint-dtw-selective','continuous-anchor-linear','continuous-anchor-dtw',
    'continuous-anchor-structural','bar-landmark-regularized'}
EXCLUDED={'nominal','offset','linear','dtw'}


def _measures():
    return [{'playback_index':i,'playback_measure_number':i+1,'written_measure_number':i+1,
             'start_time':i*2.0,'end_time':(i+1)*2.0} for i in range(4)]


def _events(offset=0.0):
    pitches=(60,62,64,65,67)
    return [{'time':offset+i*.45,'midi':pitches[i%len(pitches)]} for i in range(16)]


def test_shared_guarded_method_policy_matches_transcription_investigator():
    assert msf.SUPPORTED_ALIGNMENT_METHODS==EXPECTED
    assert ti.SUPPORTED_METHODS==EXPECTED
    assert msf.EXCLUDED_ALIGNMENT_METHODS==EXCLUDED


def test_midi_section_fit_accepts_every_guarded_method_and_rejects_simple_methods():
    for method in EXPECTED:msf.validate_alignment_method(method)
    for method in EXCLUDED:
        with pytest.raises(ValueError):msf.validate_alignment_method(method)


def test_requested_baseline_and_prepared_clock_are_both_search_hypotheses():
    report=msf.build_section_candidate_report(
        _measures(),{'piano':_events()},{'piano':_events(3.0)},
        baseline_hypotheses={
            'requested_method_baseline':lambda t:t+20.0,
            'prepared_chart_clock':lambda t:t,
        },
        cfg=msf.SectionFitConfig(search_radius_seconds=5.0,candidate_limit=5))
    assert report['search_hypotheses']==['requested_method_baseline','prepared_chart_clock']
    candidates=report['roles']['piano']['windows'][0]['candidates']
    assert candidates
    assert candidates[0]['search_hypothesis']=='prepared_chart_clock'


def test_all_guarded_methods_are_wired_and_simple_methods_are_explicitly_excluded():
    builder=(ROOT/'build_feedpak.py').read_text()
    runner=(ROOT/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    process=(ROOT/'process_gp_alignment.py').read_text()
    for method in EXPECTED:
        assert method in builder
        assert method in runner
    assert 'msf.validate_alignment_method(alignment_mode)' in process
    assert 'requested_method_baseline' in process
    assert 'prepared_chart_clock' in process
    assert '-MidiSectionFit is not available' in runner
    assert '--midi-section-fit is unavailable' in builder
