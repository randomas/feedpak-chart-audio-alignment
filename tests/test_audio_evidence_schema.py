import copy,json
import pytest
import audio_evidence_schema as aes


def test_required_identity_and_explicit_clocks():
    row=aes.make_evidence(evidence_id='e1',evidence_type='opening',detector_module='m',detector_method='f',
        clocks={'source_audio_time':1.0,'padded_audio_time':2.5,'chart_time':2.0,'mapped_audio_time':2.7},
        confidence=1.4)
    assert row['clocks']=={'source_audio_time':1.0,'padded_audio_time':2.5,'chart_time':2.0,'mapped_audio_time':2.7}
    assert row['confidence']==1.0
    assert row['diagnostic_only'] is True
    assert row['timing_changes_applied'] is False
    assert row['midi_authority_changed'] is False
    json.dumps(row)


def test_missing_values_are_not_zero_filled():
    row=aes.make_evidence(evidence_id='e1',evidence_type='x',detector_module='m',detector_method='f',
                          clocks={'source_audio_time':None})
    assert row['clocks']=={}
    assert 'confidence' not in row
    assert 'residual_seconds' not in row


def test_required_fields_are_enforced():
    with pytest.raises(ValueError):
        aes.make_evidence(evidence_id='',evidence_type='x',detector_module='m',detector_method='f')


def test_original_evidence_is_deep_copied():
    origin={'nested':{'values':[1,2]}}
    row=aes.make_evidence(evidence_id='e1',evidence_type='x',detector_module='m',detector_method='f',original=origin)
    row['original_evidence']['nested']['values'].append(3)
    assert origin=={'nested':{'values':[1,2]}}
