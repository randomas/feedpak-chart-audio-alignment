import json
import feedpak_common as fc


def test_write_json_retries_transient_permission_error(tmp_path,monkeypatch):
    target=tmp_path/'result.json';real_replace=fc.os.replace;calls={'n':0}
    def flaky(src,dst):
        calls['n']+=1
        if calls['n']<3:raise PermissionError('temporary Windows lock')
        return real_replace(src,dst)
    monkeypatch.setattr(fc.os,'replace',flaky)
    monkeypatch.setattr(fc.time,'sleep',lambda delay:None)
    fc.write_json(target,{'ok':True},replace_attempts=4)
    assert calls['n']==3
    assert json.loads(target.read_text(encoding='utf-8'))=={'ok':True}
    assert not (tmp_path/'result.json.tmp').exists()


def test_write_json_raises_after_bounded_permission_retries(tmp_path,monkeypatch):
    target=tmp_path/'result.json';calls={'n':0}
    def locked(src,dst):
        calls['n']+=1
        raise PermissionError('persistent Windows lock')
    monkeypatch.setattr(fc.os,'replace',locked)
    monkeypatch.setattr(fc.time,'sleep',lambda delay:None)
    try:fc.write_json(target,{'ok':False},replace_attempts=3)
    except PermissionError:pass
    else:raise AssertionError('PermissionError should be re-raised')
    assert calls['n']==3
