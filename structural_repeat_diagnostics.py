"""v16.6A.3 diagnostic-only phrase-level repeat correspondence."""
from __future__ import annotations
from collections import Counter
from typing import Mapping, Sequence, Any
import hashlib

VERSION="v16.6B-phrase-repeat-cleanup"


def _time(row):
    for key in ("time","t_gp","score_time","audio_time","start_time","t"):
        if isinstance(row,Mapping) and row.get(key) is not None:return float(row[key])
    return None


def _symbols(row):
    values=[]
    for key in ("pitches","pieces","pitch","midi","p","canonical_drum_piece"):
        value=row.get(key) if isinstance(row,Mapping) else None
        if value is None:continue
        values.extend(value if isinstance(value,(list,tuple,set)) else [value])
    result=[]
    for value in values:
        try:result.append(int(value))
        except (TypeError,ValueError):result.append(int(hashlib.sha256(str(value).encode()).hexdigest()[:8],16)%4096)
    return sorted(set(result))


def _measure(row,index):
    start=float(row.get("start",row.get("start_time",row.get("t_gp",0.0))))
    end=row.get("end",row.get("end_time"));end=float(end if end is not None else start+float(row.get("duration",1.0)))
    return {"playback_index":int(row.get("playback_index",index)),
      "playback_measure_number":int(row.get("playback_measure_number",row.get("measure",index+1))),
      "written_measure_number":int(row.get("written_measure_number",row.get("source_bar",row.get("written_measure",index+1)))),
      "source_occurrence":int(row.get("occurrence",row.get("source_occurrence",1)) or 1),
      "playback_segment":int(row.get("playback_segment",row.get("playback_segment_index",0)) or 0),
      "start":start,"end":end,"duration":end-start,"time_signature":row.get("time_signature",row.get("ts")),
      "repeat_iteration":row.get("repeat_iteration",row.get("occurrence")),"section":row.get("section"),
      "navigation":row.get("navigation")}


def _signature(events,start,end):
    duration=max(1e-6,end-start);items=[]
    for row in events or []:
        t=_time(row)
        if t is None or not start-1e-9<=t<end-1e-9:continue
        items.append((round((t-start)/duration,3),tuple(_symbols(row))))
    items.sort();onsets=[x[0] for x in items];symbols=[list(x[1]) for x in items]
    return {"event_count":len(items),"relative_onsets":onsets,"symbol_sets":symbols,"observable":len(items)>=4}


def _similarity(a,b):
    if not a.get("observable") or not b.get("observable"):return 0.0
    ao,bo=a["relative_onsets"],b["relative_onsets"];n=min(len(ao),len(bo))
    count=max(0.0,1-abs(len(ao)-len(bo))/max(len(ao),len(bo),1))
    rhythm=sum(max(0.0,1-abs(ao[i]-bo[i])/.14) for i in range(n))/max(len(ao),len(bo),1)
    aset={v for g in a["symbol_sets"] for v in g};bset={v for g in b["symbol_sets"] for v in g}
    symbol=len(aset&bset)/max(1,len(aset|bset)) if aset or bset else .5
    return .55*rhythm+.25*count+.20*symbol


def _inventory(measures,chart):
    out=[]
    for m in measures:
        roles={r:_signature(events,m["start"],m["end"]) for r,events in chart.items()}
        out.append({**m,"active_roles":sorted(r for r,s in roles.items() if s["event_count"]),
                    "empty":not any(s["event_count"] for s in roles.values())})
    return out


def _phrase_candidates(measures,events,role,length):
    rows=[]
    for i in range(0,len(measures)-length+1):
        start,end=measures[i]["start"],measures[i+length-1]["end"]
        sig=_signature(events,start,end)
        if sig["observable"]:rows.append({"role":role,"phrase_measures":length,"start_index":i,"end_index":i+length-1,
            "start_measure":measures[i]["playback_measure_number"],"end_measure":measures[i+length-1]["playback_measure_number"],
            "start_time":start,"end_time":end,"duration":end-start,"signature":sig})
    return rows


def _repeat_runs(phrases,threshold=.84):
    by={(x["role"],x["phrase_measures"],x["start_index"]):x for x in phrases};runs=[]
    for phrase in phrases:
        step=phrase["phrase_measures"]
        if phrase["start_index"]>=step and (phrase["role"],step,phrase["start_index"]-step) in by:continue
        occurrences=[phrase];index=phrase["start_index"]+step
        while (phrase["role"],step,index) in by:
            nxt=by[(phrase["role"],step,index)]
            if _similarity(phrase["signature"],nxt["signature"])<threshold:break
            occurrences.append(nxt);index+=step
        if len(occurrences)>=2:
            runs.append({"run_id":f"{phrase['role']}-{step}m-run-{len(runs)+1}","role":phrase["role"],
              "phrase_measures":step,"chart_occurrence_count":len(occurrences),"chart_start":occurrences[0]["start_time"],
              "chart_end":occurrences[-1]["end_time"],"chart_start_measure":occurrences[0]["start_measure"],
              "chart_end_measure":occurrences[-1]["end_measure"],"phrase_duration":phrase["duration"],
              "signature":phrase["signature"],"chart_occurrences":[{"ordinal":i+1,"start":x["start_time"],"end":x["end_time"]} for i,x in enumerate(occurrences)]})
    return runs


def _midi_matches(run,events,threshold=.74):
    events=sorted(events or [],key=lambda x:_time(x) or 0);times=sorted(set(t for x in events if (t:=_time(x)) is not None))
    duration=run["phrase_duration"];raw=[]
    for t in times:
        sig=_signature(events,t,t+duration);score=_similarity(run["signature"],sig)
        if score>=threshold:raw.append({"start":t,"end":t+duration,"score":score})
    # Build chains from the unsuppressed hypotheses. A greedy NMS can choose a
    # high-scoring interior onset (for example 1.5 phrases later) and destroy
    # otherwise exact ordinal spacing. Joint phrase spacing decides first.
    raw.sort(key=lambda x:x["start"]);chains=[]
    for seed in raw:
        chain=[seed];target=seed["start"]+duration
        while True:
            choices=[x for x in raw if x["start"]>chain[-1]["start"] and abs(x["start"]-target)<=duration*.18]
            if not choices:break
            nxt=max(choices,key=lambda x:(x["score"],-abs(x["start"]-target)))
            chain.append(nxt);target=nxt["start"]+duration
        chains.append(chain)
    best=max(chains,key=lambda c:(len(c),sum(x["score"] for x in c),-c[0]["start"]),default=[])
    kept=[]
    for x in sorted(raw,key=lambda x:(-x["score"],x["start"])):
        if any(abs(x["start"]-y["start"])<duration*.65 for y in kept):continue
        kept.append(x)
    kept.sort(key=lambda x:x["start"])
    return best,kept


def _boundary_score(run,chart_events,midi_events,chain):
    if not chain:return 0.0
    duration=run["phrase_duration"];role=run["role"]
    cpre=_signature(chart_events.get(role) or [],run["chart_start"]-duration,run["chart_start"])
    cpost=_signature(chart_events.get(role) or [],run["chart_end"],run["chart_end"]+duration)
    mpre=_signature(midi_events.get(role) or [],chain[0]["start"]-duration,chain[0]["start"])
    mpost=_signature(midi_events.get(role) or [],chain[-1]["end"],chain[-1]["end"]+duration)
    scores=[x for x in (_similarity(cpre,mpre),_similarity(cpost,mpost)) if x>0]
    return sum(scores)/len(scores) if scores else 0.0


def build(playback_measures:Sequence[Mapping[str,Any]],chart_events_by_role:Mapping[str,Sequence[Mapping[str,Any]]],
          midi_events_by_role:Mapping[str,Sequence[Mapping[str,Any]]]):
    measures=[_measure(x,i) for i,x in enumerate(playback_measures)];inventory=_inventory(measures,chart_events_by_role)
    phrases=[]
    for role,events in sorted(chart_events_by_role.items()):
        for length in (2,4):phrases.extend(_phrase_candidates(measures,events,role,length))
    runs=_repeat_runs(phrases);corr=[];proposals=[]
    for run in runs:
        chain,all_matches=_midi_matches(run,midi_events_by_role.get(run["role"]) or [])
        confirmed=len(chain);expected=run["chart_occurrence_count"]
        boundary=_boundary_score(run,chart_events_by_role,midi_events_by_role,chain)
        probable=min(max(0,len(all_matches)-confirmed),max(0,expected-confirmed));unresolved=max(0,expected-confirmed-probable)
        if not confirmed:classification="inconclusive"
        elif confirmed==expected:classification="matching_phrase_run"
        elif confirmed>expected:
            classification="additional_midi_occurrences" if boundary>=.45 else "multiple_possible_run_instances"
        elif confirmed+probable>=expected:classification="probable_complete_run"
        elif confirmed==expected-1:classification="possible_extra_chart_or_missing_midi_phrase"
        else:classification="phrase_count_mismatch"
        confidence=(sum(x["score"] for x in chain)/len(chain) if chain else 0.0)
        confidence=.8*confidence+.2*boundary
        row={"run_id":run["run_id"],"role":run["role"],"phrase_measures":run["phrase_measures"],
          "chart_span":[run["chart_start"],run["chart_end"]],"chart_occurrence_count":expected,
          "confirmed_midi_occurrences":confirmed,"probable_additional_occurrences":probable,
          "unresolved_occurrences":unresolved,"classification":classification,"boundary_context_score":round(boundary,6),
          "joint_confidence":round(confidence,6),"excess_confirmed_occurrences":max(0,confirmed-expected),
          "midi_occurrences":[{**x,"ordinal":i+1} for i,x in enumerate(chain)],"diagnostic_only":True}
        corr.append(row)
        if classification=="matching_phrase_run" and confidence>=.76 and boundary>=.45:
            proposals.append({"proposal_type":"phrase_run_ordinal_constraint","run_id":run["run_id"],"role":run["role"],
              "authority":"diagnostic_only","timing_changes_applied":False})
        elif classification not in ("matching_phrase_run","inconclusive"):
            proposals.append({"proposal_type":"phrase_structure_review","run_id":run["run_id"],"role":run["role"],
              "classification":classification,"authority":"diagnostic_only","timing_changes_applied":False})
    # Consolidate shifted, overlapping hypotheses for one role and phrase level.
    # They remain alternatives within one structural family, not independent evidence.
    clusters=[]
    for row in sorted(corr,key=lambda x:(x["role"],x["phrase_measures"],x["chart_span"][0])):
        eligible=[c for c in clusters if c["role"]==row["role"] and c["phrase_measures"]==row["phrase_measures"]
                  and min(c["chart_span"][1],row["chart_span"][1])-max(c["chart_span"][0],row["chart_span"][0])>0]
        if eligible:
            cluster=max(eligible,key=lambda c:min(c["chart_span"][1],row["chart_span"][1])-max(c["chart_span"][0],row["chart_span"][0]))
            cluster["member_run_ids"].append(row["run_id"]);cluster["alternative_phase_origin_count"]+=1
            cluster["chart_span"]=[min(cluster["chart_span"][0],row["chart_span"][0]),max(cluster["chart_span"][1],row["chart_span"][1])]
            if (row["boundary_context_score"],row["joint_confidence"])>(cluster["selected_boundary_context_score"],cluster["selected_joint_confidence"]):
                cluster.update(selected_run_id=row["run_id"],selected_boundary_context_score=row["boundary_context_score"],
                               selected_joint_confidence=row["joint_confidence"],selected_classification=row["classification"])
        else:
            clusters.append({"cluster_id":f"{row['role']}-{row['phrase_measures']}m-cluster-{len(clusters)+1}",
              "role":row["role"],"phrase_measures":row["phrase_measures"],"chart_span":list(row["chart_span"]),
              "member_run_ids":[row["run_id"]],"alternative_phase_origin_count":1,"selected_run_id":row["run_id"],
              "selected_boundary_context_score":row["boundary_context_score"],"selected_joint_confidence":row["joint_confidence"],
              "selected_classification":row["classification"],"diagnostic_only":True})

    # Hierarchical cross-role corroboration permits compatible 2m/4m evidence.
    # Boundaries must align within one shorter phrase and structural conclusions must be compatible.
    compatible={"matching_phrase_run","probable_complete_run"}
    groups=[];used=set()
    for i,row in enumerate(corr):
        if row["classification"] not in compatible:continue
        members=[row]
        for peer in corr[i+1:]:
            if peer["role"]==row["role"] or peer["classification"] not in compatible:continue
            shorter=min((row["chart_span"][1]-row["chart_span"][0])/max(1,row["chart_occurrence_count"]),
                        (peer["chart_span"][1]-peer["chart_span"][0])/max(1,peer["chart_occurrence_count"]))
            overlap=min(row["chart_span"][1],peer["chart_span"][1])-max(row["chart_span"][0],peer["chart_span"][0])
            start_delta=abs(row["chart_span"][0]-peer["chart_span"][0]);end_delta=abs(row["chart_span"][1]-peer["chart_span"][1])
            ratio=max(row["phrase_measures"],peer["phrase_measures"])/min(row["phrase_measures"],peer["phrase_measures"])
            hierarchical=ratio in (1,2) and overlap>0 and start_delta<=shorter and end_delta<=shorter
            if hierarchical:members.append(peer)
        roles=sorted({x["role"] for x in members})
        key=tuple(sorted(x["run_id"] for x in members))
        if len(roles)>=2 and key not in used:
            used.add(key);groups.append({"group_id":f"phrase-corroboration-{len(groups)+1}","run_ids":list(key),"roles":roles,
              "phrase_measures":sorted({x["phrase_measures"] for x in members}),
              "relationship":"hierarchical" if len({x["phrase_measures"] for x in members})>1 else "same_level",
              "chart_span":[max(x["chart_span"][0] for x in members),min(x["chart_span"][1] for x in members)],
              "classification":"compatible_complete_structure","diagnostic_only":True})
    counts=Counter(x["classification"] for x in corr)
    compact=[{k:m.get(k) for k in ("playback_index","playback_measure_number","written_measure_number","source_occurrence",
      "playback_segment","start","end","duration","time_signature","repeat_iteration","section","navigation","active_roles","empty")} for m in inventory]
    return {"version":VERSION,"diagnostic_only":True,"timing_changes_applied":False,"selection_behavior_changed":False,
      "summary":{"playback_measure_count":len(measures),"phrase_candidate_count":len(phrases),"phrase_repeat_run_count":len(runs),
        "correspondence_count":len(corr),"hypothesis_cluster_count":len(clusters),"cross_role_group_count":len(groups),"proposal_count":len(proposals),
        "classification_counts":dict(counts)},"playback_structure_inventory":compact,
      "phrase_repeat_runs":[{k:v for k,v in x.items() if k!="signature"} for x in runs],"phrase_run_correspondences":corr,
      "overlapping_hypothesis_clusters":clusters,"cross_role_corroboration":groups,"structural_anchor_proposals":proposals,
      "notes":["Two- and four-measure phrases are primary; single measures are not authoritative.",
               "Confirmed, probable, unresolved, and excess occurrences are reported separately.",
               "Shifted overlapping runs are clustered as alternative phase origins.",
               "Two- and four-measure roles may corroborate hierarchically when boundaries agree.",
               "All outputs remain diagnostic-only."]}
