# MIDI Section Phrase-Repeat Cleanup v16.6B

## Scope

v16.6B cleanup remains diagnostic-only. It does not change candidate generation, scoring, role paths, fusion, materialization, or final timing.

## Excess occurrence classification

Confirmed MIDI occurrence counts greater than chart counts are no longer labelled `probable_complete_run`.

They are classified as:

- `additional_midi_occurrences` when boundary context is reasonably supportive;
- `multiple_possible_run_instances` when boundaries do not identify one run instance.

`excess_confirmed_occurrences` is reported explicitly. These classifications cannot emit ordinal-anchor proposals.

## Shifted-hypothesis consolidation

Overlapping runs for the same role and phrase length are grouped into one structural hypothesis cluster. Each cluster records:

- member run IDs;
- number of alternative phase origins;
- combined chart span;
- selected representative;
- representative boundary and joint-confidence values;
- representative classification.

The representative is chosen by boundary support first and joint confidence second. Alternative phase origins remain visible and are not converted into independent evidence.

## Hierarchical cross-role corroboration

Compatible two- and four-measure evidence may corroborate across roles when:

- chart spans overlap;
- starts and ends agree within one shorter phrase;
- phrase lengths are equal or related by 2:1;
- both classifications describe complete or probably complete structure.

The resulting group identifies whether the relationship is same-level or hierarchical. No role locations are averaged.

## Safety

All cleanup products remain diagnostic-only. Selection and production behavior are unchanged.

## Validation

Integrated suite: **352 passed, 1 skipped**.
