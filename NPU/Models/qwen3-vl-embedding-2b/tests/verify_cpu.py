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

def guard_tests(params):
    import io,types,numpy as np,torch
    from qwen_embedding import pyav_reader as reader
    from qwen_embedding.cpu_decode import _initialize
    from qwen_embedding import cpu_decode
    _initialize(dict(params,decode_processes=0,decode_short_side=32))
    av_open=reader.av.open;original_hint=reader._frame_count_hint;cases=[]
    class Frame:
        def __init__(self,i,corrupt=False,vary=False):
            self.is_corrupt=corrupt;self.width=64 if vary else 32;self.height=32
            self.color_primaries=2;self.color_trc=2;self.colorspace=2;self.format=types.SimpleNamespace(name='rgb24');self.i=i
        def to_ndarray(self,format):return np.full((self.height,self.width,3),self.i,dtype=np.uint8)
    active={};counts={'decode':0}
    class Container:
        def __init__(self):self.streams=types.SimpleNamespace(video=[types.SimpleNamespace(frames=active['hint'],average_rate=2.,thread_count=1)])
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def decode(self,stream):
            counts['decode']+=1
            for i in range(active['actual']):yield Frame(i,active.get('corrupt')==i,active.get('vary')==i)
    try:
        reader.av.open=lambda *a,**k:Container()
        for name,hint,actual,extra,passes in [('valid-header',8,8,{},1),('unknown-header',0,8,{},2),('overstated-header',10,8,{},3),('understated-header',6,8,{},3),('single-frame',1,1,{},1),('unselected-corrupt',8,8,{'corrupt':6},None),('unselected-dimension-change',8,8,{'vary':6},None),('empty-video',0,0,{},None)]:
            active.clear();active.update(hint=hint,actual=actual,**extra);results=[]
            for mode in ['two-pass','single-pass']:
                counts['decode']=0;cpu_decode._extractor.parameters=dict(cpu_decode._extractor.parameters,decode_mode=mode)
                try:
                    v=cpu_decode._extractor.prepare(b'fixture');result=dict(shape=list(v.pixels.shape),metadata=v.metadata,duration=v.duration,pixels=v.pixels.numpy().copy())
                except ValueError as error:result=dict(error=type(error).__name__,message=str(error))
                results.append((result,counts['decode']))
            left,right=results[0][0],results[1][0]
            assert left.keys()==right.keys(),(name,left,right)
            if 'error' in left:assert left==right,(name,left,right)
            else:
                assert np.array_equal(left.pop('pixels'),right.pop('pixels')) and left==right,name
                assert results[1][1]==passes,(name,results[1][1],passes)
            cases.append(dict(name=name,pass_result=True,decode_passes=[r[1] for r in results],rejects_invalid='error' in left))
    finally:reader.av.open=av_open;reader._frame_count_hint=original_hint
    return cases

def rgb_test():
    import numpy as np
    from qwen_embedding.portable_rgb import convert_yuv420
    rng=np.random.default_rng(914)
    for colorspace in [0,1,2,3,4,5,6,7,9]:
        for full in [False,True]:
            x=rng.integers(0,256,(3,48,64),dtype=np.uint8)
            a=convert_yuv420(x,32,64,colorspace,full,'numpy')
            b=convert_yuv420(x,32,64,colorspace,full,'cpp')
            assert np.array_equal(a,b)
    return 18

def real_video():
    import av,numpy as np
    from qwen_embedding import cpu_decode as cd
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'synthetic.mp4'
        with av.open(str(p),'w') as c:
            s=c.add_stream('libx264',rate=4);s.width=64;s.height=32;s.pix_fmt='yuv420p'
            for i in range(16):
                f=av.VideoFrame.from_ndarray(np.full((32,64,3),i*10,dtype=np.uint8),format='rgb24')
                for packet in s.encode(f):c.mux(packet)
            for packet in s.encode():c.mux(packet)
        values=[]
        for mode in ['two-pass','single-pass']:
            cd._extractor.parameters['decode_mode']=mode
            v=cd._extractor.prepare(p)
            values.append((v.pixels.numpy().copy(),v.metadata,v.duration))
        assert np.array_equal(values[0][0],values[1][0]) and values[0][1:]==values[1][1:]
    return True

if __name__=='__main__':
    result=basic()
    params={'sample_fps':1.,'max_frames':64,'frame_transport':'uint8','decode_short_side':32,'cpu_threads':1}
    result['decode_guards']=guard_tests(params)
    result['synthetic_video_exact']=real_video()
    result['rgb_exact_cases']=rgb_test()
    print(json.dumps(result,indent=2))
