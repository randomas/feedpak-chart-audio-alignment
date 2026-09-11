import difflib,json,os,re,unicodedata
ROLES={'lead_vocal','drums','piano_left','piano_right','piano_combined','guitar','bass','ignore'}
def norm(s): return re.sub(r'\s+',' ',re.sub(r'[()\[\]{}_/\\-]+',' ',unicodedata.normalize('NFKC',str(s or '')).strip().lower())).strip()
def classify_name(name,percussion=False):
 n=norm(name)
 if n in {'lead vocals','lead vocal','main vocals','main vocal','lead vox','main vox'}: return 'lead_vocal'
 if n in {'backing vocals','backing vocal','background vocals','harmony vocals','harmony vocal'}: return 'ignored_vocal'
 if n in {'drums','drum kit','main drums','kit'}: return 'drums'
 if n in {'tambourine','triangle','shaker','cowbell','aux percussion','percussion'} or percussion:return 'ignored_percussion'
 if ('piano' in n or 'keys' in n or 'keyboard' in n) and ('lh' in n.split() or 'left' in n.split()):return 'piano_left'
 if ('piano' in n or 'keys' in n or 'keyboard' in n) and ('rh' in n.split() or 'right' in n.split()):return 'piano_right'
 if n in {'piano','keys','keyboard','grand piano'}:return 'piano_combined'
 if 'bass' in n:return 'bass'
 if 'guitar' in n:return 'guitar'
 return 'unsupported'
def inventory_song(song):
 out=[]
 for i,t in enumerate(song.tracks,1):
  notes=texts=voices=0
  for m in t.measures:
   voices=max(voices,len(m.voices))
   for v in m.voices:
    for b in v.beats:notes+=len(b.notes);texts+=bool(getattr(b,'text',None))
  out.append({'index':i,'gp_number':getattr(t,'number',i),'name':t.name,'normalized_name':norm(t.name),'is_percussion':bool(t.isPercussionTrack),'midi_program':getattr(t.channel,'instrument',-1),'string_count':len(t.strings),'measure_count':len(t.measures),'voice_count':voices,'note_count':notes,'beat_text_count':texts,'automatic_role':classify_name(t.name,t.isPercussionTrack)})
 return out
def default_allocation(inv):
 first=lambda r:next((x['name'] for x in inv if x['automatic_role']==r),None)
 active=[first(x) for x in ('lead_vocal','drums','piano_left','piano_right','piano_combined')]
 guitars=[x['name'] for x in inv if x['automatic_role']=='guitar'];bass=[x['name'] for x in inv if x['automatic_role']=='bass']; active=set(x for x in active+guitars+bass if x)
 ignored=[x['name'] for x in inv if x['automatic_role'] in {'ignored_vocal','ignored_percussion'}]
 return {'lead_vocal':first('lead_vocal'),'drums':first('drums'),'piano':{'left':first('piano_left'),'right':first('piano_right'),'combined':first('piano_combined')},'guitars':guitars,'bass':bass,'ignored_tracks':ignored,'overrides':{},'unused_tracks':[x['name'] for x in inv if x['name'] not in active]}
def names(t):
 p=t.get('piano') or {};return [x for x in [t.get('lead_vocal'),t.get('drums'),p.get('left'),p.get('right'),p.get('combined')] if x]+(t.get('guitars') or [])+(t.get('bass') or [])+(t.get('ignored_tracks') or [])+list((t.get('overrides') or {}).keys())
def ensure_song_json(path,gp,inv):
 existed=os.path.isfile(path);data=json.load(open(path,encoding='utf8')) if existed else {'title':'Unknown Title','artist':'Unknown Artist','album':None,'year':None,'genres':[]}
 pr=data.setdefault('feedpak_project',{});pr.setdefault('version',1);pr.setdefault('chart',gp);tr=pr.setdefault('tracks',{})
 d=default_allocation(inv)
 for k,v in d.items():tr.setdefault(k,v)
 pr['track_inventory']=[{'index':x['index'],'name':x['name'],'automatic_role':x['automatic_role']} for x in inv]
 assigned=set(names({**tr,'unused_tracks':[]}));tr['unused_tracks']=[x['name'] for x in inv if x['name'] not in assigned]

 with open(path,'w',encoding='utf8') as f:
  json.dump(data,f,indent=2,ensure_ascii=False);f.write('\n')
 return data,not existed
def validate_song_config(data,inv):
 tr=(data.get('feedpak_project') or {}).get('tracks') or {}; avail=[x['name'] for x in inv];err=[]
 for n in names(tr):
  if n not in avail:err.append({'code':'configured_track_not_found','track':n,'suggestion':next(iter(difflib.get_close_matches(n,avail,n=1,cutoff=.55)),None)})
  elif avail.count(n)>1:err.append({'code':'configured_track_name_ambiguous','track':n})
 ov=tr.get('overrides') or {}
 for n,r in ov.items():
  if r not in ROLES:err.append({'code':'unknown_override_role','track':n,'role':r})
 pairs=[]
 for r in ('lead_vocal','drums'):
  if tr.get(r):pairs.append((tr[r],r))
 p=tr.get('piano') or {}
 for r in ('left','right','combined'):
  if p.get(r):pairs.append((p[r],'piano_'+r))
 for n in tr.get('guitars') or []:pairs.append((n,'guitar'))
 for n in tr.get('bass') or []:pairs.append((n,'bass'))
 for n,r in ov.items():
  if r!='ignore':pairs.append((n,r))
 for n in set(x[0] for x in pairs):
  rs=set(r for x,r in pairs if x==n)
  if len(rs)>1:err.append({'code':'incompatible_duplicate_assignment','track':n,'roles':sorted(rs)})
 ignored=set(tr.get('ignored_tracks') or [])|{n for n,r in ov.items() if r=='ignore'}
 for n,r in pairs:
  if n in ignored:err.append({'code':'active_track_also_ignored','track':n})
 if p.get('left') and p.get('left')==p.get('right'):err.append({'code':'piano_hands_use_same_track','track':p['left']})
 return {'valid':not err,'errors':err,'warnings':[]}
def resolved_roles(data,inv):
 tr=data['feedpak_project']['tracks'];out={x['name']:x['automatic_role'] for x in inv}
 for n in tr.get('unused_tracks') or []:out[n]='unsupported'
 for n in tr.get('ignored_tracks') or []:out[n]='ignored'
 for r in ('lead_vocal','drums'):
  if tr.get(r):out[tr[r]]=r
 p=tr.get('piano') or {}
 for r in ('left','right','combined'):
  if p.get(r):out[p[r]]='piano_'+r
 for n in tr.get('guitars') or []:out[n]='guitar'
 for n in tr.get('bass') or []:out[n]='bass'
 out.update(tr.get('overrides') or {});return out
def inspect_song(song,gp,data,inv,val):
 roles=resolved_roles(data,inv);ly=getattr(song,'lyrics',None);raw=getattr(ly,'trackChoice',None);lines=[{'line':i,'starting_measure':l.startingMeasure,'text':l.lyrics} for i,l in enumerate(getattr(ly,'lines',[]),1) if l and l.lyrics]
 tempo=[{'tick':0,'bpm':float(song.tempo),'source':'song_initial_tempo'}]
 for t in song.tracks[:1]:
  for m in t.measures:
   for v in m.voices:
    for b in v.beats:
     c=getattr(getattr(b,'effect',None),'mixTableChange',None)
     if c and c.tempo is not None:tempo.append({'tick':b.start,'measure':m.header.number,'bpm':float(c.tempo.value),'source':'gp_mix_table_change'})
 return {'version':1,'source':{'file':os.path.basename(gp),'track_count':len(inv),'measure_count':len(song.measureHeaders)},'tracks':[{**x,'final_role':roles[x['name']],'action':'process' if roles[x['name']] in {'lead_vocal','drums','guitar','bass','piano_left','piano_right','piano_combined'} else 'ignore'} for x in inv],'lyrics':{'raw_track_choice':raw,'one_based_candidate':inv[raw-1]['name'] if isinstance(raw,int) and 1<=raw<=len(inv) else None,'zero_based_candidate':inv[raw]['name'] if isinstance(raw,int) and 0<=raw<len(inv) else None,'lines':lines},'tempo_map':tempo,'validation':val}













def classify_vocal_capability(song, inventory=None, roles=None):
    inventory = inventory if inventory is not None else inventory_song(song)
    roles = roles or {x['name']:x['automatic_role'] for x in inventory}
    names=[n for n,r in roles.items() if r=='lead_vocal']
    tracks=[t for t in getattr(song,'tracks',[]) if t.name in names]
    notes=sum(len(getattr(b,'notes',[]) or []) for t in tracks for m in t.measures for v in m.voices for b in v.beats)
    timed=[{'track':t.name,'measure':mi,'tick':int(b.start),'text':str(b.text)} for t in tracks for mi,m in enumerate(t.measures,1) for v in m.voices for b in v.beats if getattr(b,'text',None) and getattr(b,'start',None) is not None]
    lines=[{'line':i,'starting_measure':int(getattr(l,'startingMeasure',1) or 1),'text_length':len(str(l.lyrics))} for i,l in enumerate(getattr(getattr(song,'lyrics',None),'lines',[]) or [],1) if l and getattr(l,'lyrics',None) and str(l.lyrics).strip()]
    starts=sorted({x['starting_measure'] for x in lines})
    if not tracks or (not notes and not timed and not lines): kind,ok,why='no_vocal_material',False,'no configured GP vocal notes or lyrics'
    elif timed: kind,ok,why='timed_beat_text',True,'timed beat.text lyric anchors are present'
    elif len(lines)>=2 and len(starts)>=2: kind,ok,why='measure_allocated_lyrics',True,'multiple lyric lines have distinct starting measures'
    elif lines: kind,ok,why='flat_untimed_lyrics',False,'song-level lyrics have no beat or recoverable measure timing'
    else: kind,ok,why='notes_only',False,'vocal notes exist without synchronized lyric text'
    return {'classification':kind,'direct_gp_lyrics_supported':ok,'reason':why,'configured_tracks':names,'vocal_note_count':notes,'timed_beat_text_count':len(timed),'timed_beat_text':timed,'nonempty_lyric_lines':lines,'distinct_starting_measures':starts}






