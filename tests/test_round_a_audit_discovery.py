import csv,json
from pathlib import Path
import audit_reports as ar

def write_json(path,data):path.write_text(json.dumps(data),encoding='utf-8')

def test_one_successful_material_produces_exactly_one_row_and_ignores_skeleton(tmp_path):
    actual=tmp_path/'01_song';actual.mkdir()
    diagnostics=tmp_path/'Diagnostics_light'/'song'/'method'/'skeleton';diagnostics.mkdir(parents=True)
    (actual/'song.feedpak').write_bytes(b'x')
    write_json(actual/'song.feedpak.method_result_envelope.json',{'method_id':'continuous-anchor-dtw','status':'complete_with_warnings','warning_count':1})
    write_json(actual/'song.feedpak.alignment_report.json',{'production_timing':{'requested_method':'continuous-anchor-dtw','selected_production_method':'continuous-anchor-dtw','timing_output':'manual-anchor-fallback','fallback_used':True}})
    write_json(actual/'song.feedpak.beatmap_report.json',{'manual_anchor_verification':{'exported_manual_anchor_status':'satisfied','maximum_exported_manual_anchor_error_ms':0}})
    write_json(actual/'song.feedpak.runtime_contract.json',{'status':'satisfied','requested':{'alignment_mode':'continuous-anchor-dtw','vocals_mode':'off','anchors':'metadata.json'},'delivered':{'method':'continuous-anchor-dtw','vocals':False}})
    write_json(diagnostics/'fake.method_result_envelope.json',{'status':'failed'})
    write_json(tmp_path/'test_material_summary.json',[{'Material':'song','Output':str(actual),'Status':'PASS','ExitCode':0}])
    out=ar.write_comparison_audit(tmp_path)
    rows=list(csv.DictReader((out/'method_scorecard.csv').open(encoding='utf-8-sig')))
    assert len(rows)==1
    row=rows[0]
    assert row['requested_method']=='continuous-anchor-dtw'
    assert row['exit_code']=='0'
    assert row['fallback_used']=='True'
    assert row['anchors_requested']=='metadata.json'
    assert row['vocals_requested']=='off' and row['vocals_delivered']=='False'
    trace=json.loads((out/'timing_route_trace.json').read_text())
    assert trace[0]['timing_output']=='manual-anchor-fallback'
