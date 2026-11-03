from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_all_method_runner_lists_every_public_method():
    text=(ROOT/'run_all_alignment_methods.ps1').read_text(encoding='utf-8-sig')
    for mode in ('nominal','offset','linear','dtw','dtw-checkpoint','checkpoint-linear',
                 'checkpoint-dtw-diagnostic','checkpoint-dtw-selective',
                 'continuous-anchor-linear','continuous-anchor-dtw',
                 'continuous-anchor-structural','bar-landmark-regularized'):
        assert f'"{mode}"' in text

def test_vocal_cache_is_reused_after_first_method():
    all_methods=(ROOT/'run_all_alignment_methods.ps1').read_text(encoding='utf-8-sig')
    corpus=(ROOT/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    assert '$methodIndex -gt 1' in all_methods
    assert '$params.ReuseVocals = $true' in all_methods
    assert '[switch]$ReuseVocals' in corpus
    assert '$arguments += "--reuse-vocals"' in corpus

def test_comparison_writes_separate_method_directories_and_summary():
    text=(ROOT/'run_all_alignment_methods.ps1').read_text(encoding='utf-8-sig')
    assert 'Join-Path $OutputRoot $method' in text
    assert 'all_methods_summary.csv' in text
    assert 'all_methods_summary.json' in text


def test_hf_token_defaults_from_environment_and_is_forwarded():
    all_methods=(ROOT/'run_all_alignment_methods.ps1').read_text(encoding='utf-8-sig')
    corpus=(ROOT/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    assert '[string]$HFToken = $env:HF_TOKEN' in all_methods
    assert '[string]$HFToken = $env:HF_TOKEN' in corpus
    assert '$params.HFToken = $HFToken' in all_methods
    assert '$arguments += @("--hf-token", $HFToken)' in corpus

def test_single_method_failure_does_not_abort_remaining_methods():
    text=(ROOT/'run_all_alignment_methods.ps1').read_text(encoding='utf-8-sig')
    assert 'throw "Method $method failed' not in text
    assert '$failedMethods.Add($method)' in text
    assert 'Continuing with the remaining methods' in text
    assert 'All requested methods were attempted' in text


def test_hf_tokens_are_redacted_from_command_and_process_logs():
    corpus=(ROOT/'run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    assert 'function Protect-SensitiveText' in corpus
    assert '$Arguments[$i - 1] -eq "--hf-token"' in corpus
    assert '$value = "<redacted>"' in corpus
    assert 'Protect-SensitiveText -Text $_.ToString() -Secrets @($HFToken)' in corpus
    assert '<redacted-hf-token>' in corpus
