import copy,json
from pathlib import Path
import midi_section_fit as msf


def fixture():
    measures=[{'playback_index':i,'playback_measure_number':i+1,'written_measure_number':i+1,
               'start_time':float(i*4),'end_time':float((i+1)*4)} for i in range(8)]
    chart={
      'bass':[{'t_gp':float(x),'midi':40+(x%5)} for x in range(0,32)],
      'guitar':[{'t_gp':float(x)+.2,'midi':60+(x%7)} for x in range(0,32)]}
    midi={role:[{'audio_time':e['t_gp']+1.0,'pitch':e['midi'],'event_type':'pitched_note','source':role}
                for e in events] for role,events in chart.items()}
    return measures,chart,midi


def canonical(report):
    out=copy.deepcopy(report)
    perf=out.get('performance') or {}
    for k in list(perf):
        if 'seconds' in k or k.startswith('role_cache_'):perf.pop(k,None)
    for role in (out.get('roles') or {}).values():
        rp=role.get('performance') or {};rp.pop('elapsed_seconds',None)
    return out


def test_role_cache_is_candidate_exact_and_warm_reuse(tmp_path):
    measures,chart,midi=fixture();cache=tmp_path/'midi_section_candidates.json'
    cold=msf.build_section_fusion_report(measures,chart,midi,candidate_cache_path=str(cache))
    # Force role-cache path by removing only the monolithic compatibility file.
    cache.unlink()
    warm=msf.build_section_fusion_report(measures,chart,midi,candidate_cache_path=str(cache))
    assert canonical(cold['candidate_report'])==canonical(warm['candidate_report'])
    cr=warm['performance']['candidate_cache']
    assert cr['role_cache_hits']==['bass','guitar']
    assert cr['role_cache_misses']==[]


def test_valid_legacy_monolithic_cache_is_accepted_and_imported(tmp_path):
    measures,chart,midi=fixture();cache=tmp_path/'midi_section_candidates.json'
    first=msf.build_section_fusion_report(measures,chart,midi,candidate_cache_path=str(cache))
    role_dir=tmp_path/'midi_section_candidates.roles'
    if role_dir.exists():
        for x in role_dir.iterdir():x.unlink()
        role_dir.rmdir()
    second=msf.build_section_fusion_report(measures,chart,midi,candidate_cache_path=str(cache))
    cr=second['performance']['candidate_cache']
    assert cr['legacy_monolithic_accepted'] is True
    assert sorted(cr['legacy_roles_imported'])==['bass','guitar']
    assert (role_dir/'bass.json').is_file() and (role_dir/'guitar.json').is_file()
    assert canonical(first['candidate_report'])==canonical(second['candidate_report'])


def test_changing_one_role_reuses_the_other_role(tmp_path):
    measures,chart,midi=fixture();cache=tmp_path/'midi_section_candidates.json'
    msf.build_section_fusion_report(measures,chart,midi,candidate_cache_path=str(cache))
    cache.unlink();midi['guitar']=midi['guitar']+[{'audio_time':40.0,'pitch':72,'event_type':'pitched_note','source':'guitar'}]
    report=msf.build_section_fusion_report(measures,chart,midi,candidate_cache_path=str(cache))
    cr=report['performance']['candidate_cache']
    assert cr['role_cache_hits']==['bass']
    assert cr['role_cache_misses']==['guitar']


def test_progress_markers_and_piano_collision_guard_are_wired():
    root=Path(__file__).resolve().parents[1]
    midi=(root/'midi_section_fit.py').read_text(encoding='utf-8')
    assert 'MIDI candidates [{role}]: window' in midi
    assert 'legacy_monolithic_import' in midi
    builder=(root/'build_feedpak.py').read_text(encoding='utf-8')
    assert 'has_piano_arrangement' in builder
    assert 'piano_path=None' in builder
