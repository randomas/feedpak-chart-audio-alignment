# v16.7K3.1.1 MIDI-section stage progress

## Scope

This patch adds immediate, flushed console output around cached candidate loading, monotonic role-path selection, and cross-role fusion. It changes no cache fingerprint, candidate, score, path, control, region, authority, materialization, or final timing.

## Messages

A monolithic hit is printed immediately, including load time and imported roles. Role-cache hit/miss summaries are printed before path selection. Path selection reports start and completion. Fusion reports its layer and role counts before work begins, emits a heartbeat approximately every 15 seconds, and reports completion counts and elapsed time.

The heartbeat includes completed layer position, block position, selection-call count, failed selection count, elapsed time, and estimated remaining time. All output uses `flush=True` for immediate PowerShell display.

## Equivalence

Fusion progress counters are diagnostic additions only. Tests call fusion with effectively disabled and always-on heartbeat intervals and compare complete selection output after removing only `progress_diagnostics`.
