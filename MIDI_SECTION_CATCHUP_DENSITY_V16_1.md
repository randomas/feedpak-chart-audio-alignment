# MIDI section catch-up density v16.1

This release replaces raw catch-up pair counts with correction episodes and length-normalized allowances. Overlapping pair warnings that share an edge are one episode. Allowances use unique covered playback measures: conservative 1/24, permissive 1/16, diagnostic-unsafe 1/8, with a one-episode minimum grace.

The policy changes evidence tiering only. Search grids, exact Phase-1 caching, candidate scoring, path selection, fusion geometry, materialized controls, and hard monotonicity constraints are unchanged.

The motivating DSMN pattern `0.93 -> 1.25 -> 0.9325` is one central local-correction episode, not two independent failures.
