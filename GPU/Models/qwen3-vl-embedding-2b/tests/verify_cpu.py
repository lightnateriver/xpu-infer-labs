"""CPU-only checks; generated/temporary data, no checkpoint or accelerator."""
import hashlib,json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))

def basic():
    from qwen_embedding.cpu_decode import read_tar_payload
    from qwen_embedding.video import QwenVLLMExtractor
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'payload';p.write_bytes(b'prefix'+b'video'+b'suffix')
        h=hashlib.sha256(b'video').hexdigest()
        assert read_tar_payload(p,6,5,h)==b'video'
        for offset,size,sha in [(5,5,h),(6,100,h),(6,5,'0'*64)]:
            try:read_tar_payload(p,offset,size,sha)
            except ValueError:pass
            else:raise AssertionError('Invalid payload accepted')
    for x in [np.zeros((1,2048)),np.full((1,2048),np.nan),np.ones((1,10))]:
        try:QwenVLLMExtractor._normalize(x)
        except ValueError:pass
        else:raise AssertionError('Invalid vector accepted')
    return {'payload_guards':3,'invalid_vector_guards':3}

if __name__=='__main__':print(json.dumps(basic()))
