from pathlib import Path
import inspect
import process_gp_alignment as pga
import build_feedpak as bf


def test_processor_default_is_split_classic():
    assert inspect.signature(pga.playable_keyboard_from_notation).parameters['piano_rendering_policy'].default=='split-classic'
    assert inspect.signature(pga.process).parameters['piano_rendering_policy'].default=='split-classic'


def test_builder_default_is_split_classic():
    assert inspect.signature(bf.build).parameters['piano_rendering_policy'].default=='split-classic'
    assert inspect.signature(bf.run_gp_script).parameters['piano_rendering_policy'].default=='split-classic'


def test_runner_default_is_split_classic():
    text=Path('run_all_test_materials.ps1').read_text(encoding='utf-8-sig')
    assert '[string]$PianoRenderingPolicy = "split-classic"' in text
