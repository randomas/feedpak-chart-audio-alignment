from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np


def _num(v, label):
    try:
        v = float(v)
    except (TypeError, ValueError):
        raise ValueError(f'{label} must be numeric')
    if not math.isfinite(v):
        raise ValueError(f'{label} must be finite')
    return v


def _final_bar_end_time(bar):
    """Return an explicit nominal end time for a final playback bar, if supplied."""
    for key in ('end_t_gp', 't_end_gp', 'bar_end_t_gp', 'end_time_gp'):
        value = bar.get(key)
        if value is not None:
            value = _num(value, key)
            if value > float(bar['t_gp']):
                return value
    duration = bar.get('duration_gp')
    if duration is not None:
        duration = _num(duration, 'duration_gp')
        if duration > 0:
            return float(bar['t_gp']) + duration
    return None


def load_and_resolve(path, bars, padding_added=0.0):
    if not path:
        return [], {'status': 'not_configured', 'anchors': []}
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    cfg = payload.get('alignment', payload)
    default = str(cfg.get('manual_anchor_time_basis', cfg.get('time_reference', 'original'))).lower()
    if default == 'source':
        default = 'original'
    if default not in ('original', 'padded'):
        raise ValueError('time basis must be original or padded')
    rows = cfg.get('manual_anchors', cfg.get('anchors', [])) or []
    out = []
    for i, raw in enumerate(rows, 1):
        score = raw.get('score') or {}
        audio = raw.get('audio') or {
            'time': raw.get('audio_time'),
            'time_basis': raw.get('time_reference'),
        }
        basis = str(audio.get('time_basis') or default).lower()
        basis = 'original' if basis == 'source' else basis
        if basis not in ('original', 'padded'):
            raise ValueError(f'anchor {i}: invalid time basis')

        if 'playback_bar' in score:
            matches = [b for b in bars if int(b['measure']) == int(score['playback_bar'])]
        elif 'source_bar' in score:
            matches = [b for b in bars if int(b.get('source_bar', b['measure'])) == int(score['source_bar'])]
            occ = score.get('occurrence')
            if occ is None and len(matches) > 1:
                raise ValueError('repeated source_bar requires occurrence')
            if occ is not None:
                matches = [b for b in matches if int(b.get('occurrence', 1)) == int(occ)]
        else:
            matches = [b for b in bars if int(b['measure']) == int(raw.get('measure', 0))]
        if len(matches) != 1:
            raise ValueError(f'anchor {i}: score position resolves to {len(matches)} bars')

        b = matches[0]
        ordered = sorted(bars, key=lambda x: int(x['measure']))
        j = ordered.index(b)
        beat = _num(score.get('beat', raw.get('beat', 1)), 'beat')
        frac = _num(score.get('beat_fraction', 0), 'beat_fraction')
        if frac < 0 or frac >= 1:
            raise ValueError(f'anchor {i}: beat_fraction must be >= 0 and < 1')
        pos = beat - 1 + frac
        meter = max(1, int(b.get('ts_num', 4)))
        if pos < 0 or pos >= meter:
            raise ValueError(f'anchor {i}: beat outside bar meter')

        bar_start = float(b['t_gp'])
        if j + 1 < len(ordered):
            bar_end = float(ordered[j + 1]['t_gp'])
        else:
            # Beat 1 of the final playback bar is exactly its known start and
            # therefore does not require a following bar boundary.
            if abs(pos) <= 1e-12:
                bar_end = None
            else:
                bar_end = _final_bar_end_time(b)
                if bar_end is None:
                    raise ValueError(
                        f'anchor {i}: final-bar positions after beat 1 require '
                        'an explicit final-bar end time or duration'
                    )

        nominal = bar_start if bar_end is None else bar_start + pos / meter * (bar_end - bar_start)
        supplied = _num(audio.get('time'), f'anchor {i} audio time')
        mode = str(raw.get('mode', 'exact')).lower()
        if supplied < 0 or mode not in ('exact', 'search'):
            raise ValueError(f'anchor {i}: invalid audio time or mode')
        padded = supplied + padding_added if basis == 'original' else supplied
        out.append({
            'id': raw.get('id') or f'anchor-{i}',
            'label': raw.get('label'),
            'mode': mode,
            'nominal_time': nominal,
            'padded_audio_time': padded,
            'original_audio_time': supplied if basis == 'original' else supplied - padding_added,
            'supplied_audio_time': supplied,
            'time_basis': basis,
            'playback_bar': int(b['measure']),
            'source_bar': int(b.get('source_bar', b['measure'])),
            'occurrence': int(b.get('occurrence', 1)),
            'beat': beat,
            'search': raw.get('search') or {},
        })

    out.sort(key=lambda x: x['nominal_time'])
    for a, b in zip(out, out[1:]):
        if b['nominal_time'] <= a['nominal_time'] or b['padded_audio_time'] <= a['padded_audio_time']:
            raise ValueError('anchors must increase in score and audio time')
    return out, {
        'status': 'resolved',
        'default_time_basis': default,
        'padding_added_seconds': padding_added,
        'anchors': out,
    }


def assess_anchors(anchors, baseline, bars, padding_added=0.0):
    report = []
    for a in anchors:
        expected = float(baseline(a['nominal_time']))
        residual = a['padded_audio_time'] - expected
        radius = int(a['search'].get('score_radius_bars', 2))
        nearby = []
        for b in bars:
            if abs(int(b['measure']) - a['playback_bar']) <= radius:
                r = a['padded_audio_time'] - float(baseline(float(b['t_gp'])))
                nearby.append({
                    'playback_bar': int(b['measure']),
                    'residual_seconds': round(r, 6),
                    'absolute_residual_seconds': round(abs(r), 6),
                })
        nearby.sort(key=lambda x: x['absolute_residual_seconds'])
        best = nearby[0] if nearby else None
        alt = (
            a['supplied_audio_time']
            if a['time_basis'] == 'original'
            else a['supplied_audio_time'] + padding_added
        ) - expected
        warnings = []
        if best and best['playback_bar'] != a['playback_bar'] and best['absolute_residual_seconds'] + .1 < abs(residual):
            warnings.append('nearby_bar_fits_better')
        if abs(alt) + .1 < abs(residual):
            warnings.append('possible_time_basis_error')
        confidence = (
            'high' if abs(residual) <= .1 else
            'moderate' if abs(residual) <= .25 else
            'warning' if abs(residual) <= .6 else
            'suspicious'
        )
        report.append({
            **a,
            'baseline_prediction': round(expected, 6),
            'residual_seconds': round(residual, 6),
            'confidence': confidence,
            'warnings': warnings,
            'best_nearby_bar': best,
            'alternate_time_basis_residual_seconds': round(alt, 6),
        })
    exact = [x for x in report if x['mode'] == 'exact']
    segments = []
    for a, b in zip(exact, exact[1:]):
        scale = (b['padded_audio_time'] - a['padded_audio_time']) / (b['nominal_time'] - a['nominal_time'])
        segments.append({
            'from': a['id'],
            'to': b['id'],
            'implied_scale': round(scale, 8),
            'status': 'plausible' if .8 <= scale <= 1.2 else 'suspicious',
        })
    return {
        'version': 1,
        'anchors': report,
        'segments': segments,
        'exact_anchor_count': len(exact),
        'search_anchor_count': len(report) - len(exact),
    }


def apply_exact_anchors(baseline, assessment, bars):
    exact = [x for x in assessment.get('anchors', []) if x['mode'] == 'exact']
    if not exact:
        return baseline, {
            'status': 'no_exact_anchors',
            'timing_changes_applied': False,
            'points': [],
        }
    start = float(bars[0]['t_gp'])
    end = float(bars[-1]['t_gp'])
    raw = (
        [(start, float(baseline(start)), 'baseline_start')]
        + [(x['nominal_time'], x['padded_audio_time'], x['id']) for x in exact]
        + [(end, float(baseline(end)), 'baseline_end')]
    )
    ded = {}
    for x, y, kind in raw:
        if x not in ded or not kind.startswith('baseline_'):
            ded[x] = (y, kind)
    pts = sorted((x, y, kind) for x, (y, kind) in ded.items())
    if any(b[1] <= a[1] for a, b in zip(pts, pts[1:])):
        raise ValueError('manual anchors conflict with baseline endpoints')
    xs = np.array([x for x, _, _ in pts])
    ys = np.array([y for _, y, _ in pts])

    def warp(t):
        value = float(t)
        if value < xs[0] or value > xs[-1]:
            return float(baseline(value))
        return float(np.interp(value, xs, ys))

    return warp, {
        'status': 'applied',
        'timing_changes_applied': True,
        'points': [
            {'nominal_time': x, 'padded_audio_time': y, 'kind': kind}
            for x, y, kind in pts
        ],
    }


def constrain_warp_to_exact_anchors(method_warp, assessment, bars):
    """Preserve a method warp's local shape while making exact anchors immutable.

    A piecewise-linear residual correction is applied between method endpoints and
    exact manual anchors. This is intentionally a finalization layer: it can be
    applied after DTW, MIDI, structural, or section materialization without
    discarding the method's internal timing shape.
    """
    exact = sorted(
        [x for x in assessment.get('anchors', []) if x.get('mode') == 'exact'],
        key=lambda x: float(x['nominal_time'])
    )
    if not exact:
        return method_warp, {'status':'no_exact_anchors','timing_changes_applied':False,
                            'controls':[],'intervals':[]}
    start=float(bars[0]['t_gp']); end=float(bars[-1]['t_gp'])
    controls=[{'chart_time':start,'target_time':float(method_warp(start)),'kind':'method_start','id':'method_start'}]
    controls += [{'chart_time':float(x['nominal_time']),'target_time':float(x['padded_audio_time']),
                  'kind':'manual_exact','id':x.get('id')} for x in exact]
    controls += [{'chart_time':end,'target_time':float(method_warp(end)),'kind':'method_end','id':'method_end'}]
    ded={}
    for row in controls:
        x=row['chart_time']
        if x not in ded or row['kind']=='manual_exact': ded[x]=row
    controls=sorted(ded.values(),key=lambda x:x['chart_time'])
    if any(b['target_time'] <= a['target_time'] for a,b in zip(controls,controls[1:])):
        raise ValueError('manual anchors conflict with final method map endpoints or each other')
    xs=np.array([x['chart_time'] for x in controls],dtype=float)
    residuals=np.array([x['target_time']-float(method_warp(x['chart_time'])) for x in controls],dtype=float)
    def warp(t):
        value=float(t)
        if value < xs[0] or value > xs[-1]: return float(method_warp(value))
        return float(method_warp(value) + np.interp(value,xs,residuals))
    intervals=[]
    for a,b in zip(controls,controls[1:]):
        intervals.append({'from':a['id'],'to':b['id'],'chart_start':a['chart_time'],
                          'chart_end':b['chart_time'],'residual_start_seconds':
                          a['target_time']-float(method_warp(a['chart_time'])),
                          'residual_end_seconds':b['target_time']-float(method_warp(b['chart_time']))})
    # Finalization must never trade anchor accuracy for invalid beat ordering.
    sample_times=sorted(set([float(x['t_gp']) for x in bars]+list(xs)))
    sample_values=[warp(x) for x in sample_times]
    if any(b <= a for a,b in zip(sample_values,sample_values[1:])):
        fallback,fallback_report=apply_exact_anchors(method_warp,assessment,bars)
        fallback_report.update({'status':'fallback_non_monotonic_shape_correction',
                                'strategy':'anchored_linear_fallback',
                                'rejected_strategy':'shape_preserving_residual'})
        return fallback,fallback_report
    return warp, {'status':'applied','timing_changes_applied':True,'strategy':'shape_preserving_residual',
                  'controls':controls,'intervals':intervals,'monotonic_sample_count':len(sample_times)}


def verify_exported_beatmap(beatmap, assessment, tolerance_seconds=.001):
    """Verify exact anchors from exported controls; use grid interpolation only for legacy Beatmaps."""
    controls={str(x.get('id')):x for x in (beatmap.get('timing_controls') or [])
              if str(x.get('mode','exact')).casefold()=='exact'}
    points=[]
    for row in beatmap.get('beats',[]) or []:
        timing=row.get('timing') or {}
        if timing.get('score_time') is not None and timing.get('audio_time') is not None:
            points.append((float(timing['score_time']),float(timing['audio_time'])))
    points=sorted(set(points))
    anchors=[x for x in assessment.get('anchors',[]) if x.get('mode')=='exact']
    checks=[]
    xs=np.array([x for x,_ in points]) if points else None
    ys=np.array([y for _,y in points]) if points else None
    for row in anchors:
        aid=str(row.get('id')); chart=float(row['nominal_time']); target=float(row['padded_audio_time'])
        control=controls.get(aid)
        if control is not None:
            mapped=float(control['audio_time']); source='timing_control'
            score_error=abs(float(control['score_time'])-chart)
        elif points:
            mapped=float(np.interp(chart,xs,ys)); source='beat_grid_interpolation'; score_error=0.0
        else:
            continue
        error=abs(mapped-target)
        checks.append({'anchor_id':row.get('id'),'chart_time':chart,
                       'target_audio_time':target,'exported_audio_time':mapped,
                       'error_ms':error*1000.0,'score_time_error_ms':score_error*1000.0,
                       'verification_source':source,
                       'satisfied':error<=tolerance_seconds and score_error<=tolerance_seconds})
    status='not_supplied' if not anchors else ('unavailable' if len(checks)!=len(anchors) else
            ('satisfied' if all(x['satisfied'] for x in checks) else 'export_failed'))
    return {'manual_anchor_count':len(anchors),'exported_manual_anchor_status':status,
            'maximum_exported_manual_anchor_error_ms':max((x['error_ms'] for x in checks),default=0.0),
            'exported_manual_anchor_conflicts':[x for x in checks if not x['satisfied']],
            'anchors':checks}
