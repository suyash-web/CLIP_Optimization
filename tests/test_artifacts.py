import json
from pathlib import Path
from src.common import publish, artifact_ok, sha256

def test_artifacts(tmp_path):
    src=tmp_path/'input.txt';src.write_text('test artifact')
    publish(tmp_path,'phase2_artifacts/item.txt',src)
    assert artifact_ok(tmp_path,'phase2_artifacts/item.txt')
    (tmp_path/'phase2_artifacts/item.txt').write_text('corrupted')
    assert not artifact_ok(tmp_path,'phase2_artifacts/item.txt')
