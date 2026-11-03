# Phase 2b.1 and 2b.2 exact preparation cache

Phase 2b.1 adds pipeline-wide stage timers. Phase 2b.2 adds exact prepared-stem and AlphaTab-products caches, while retaining the exact persistent MIDI-section candidate cache.

No audio features, baseline decisions, final timing, warped arrangements, manifests, or FeedPak archives are reused. All caches use canonical SHA-256 fingerprints and atomic manifests. Invalid caches cause normal recomputation.

Validation: full test suite passes, including exact cache hit, invalidation, missing-artifact, and timer-wiring tests.
