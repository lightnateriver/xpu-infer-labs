"""Bounded-memory sequential reader for mixed SJTU codecs, including AV1.

Both passes validate all frames. Only requested RGB frames are retained.
"""
import av
import numpy as np

class Batch:
 def __init__(self, value): self.value=value
 def asnumpy(self): return self.value

class FrameCountHintMismatch(ValueError):
 pass

def _frame_count_hint(stream):
 return int(stream.frames or 0)

class VideoReader:
 def __init__(self, source, num_threads=1, rgb_mode="native", rgb_implementation="numpy", count_from_header=False):
  if rgb_mode not in ("native","x86-compatible"):raise ValueError("Unknown RGB mode")
  if rgb_implementation not in ("numpy","cpp"):raise ValueError("Unknown RGB implementation")
  self.rgb_mode=rgb_mode;self.rgb_implementation=rgb_implementation
  self.source=source;self.threads=num_threads;self.count=0;self.shape=None;self._count_hint_used=False
  with self._open() as c:
   if not c.streams.video: raise ValueError('No video stream')
   v=c.streams.video[0];v.thread_count=self.threads
   self.fps=float(v.average_rate or 0)
   if count_from_header:
    hint=_frame_count_hint(v)
    if hint>0:
     self.count=hint;self._count_hint_used=True;return
   for f in c.decode(v):
    self._validate(f);self.count+=1
  if self.count<1: raise ValueError('No decoded frames')
 def _open(self):
  if hasattr(self.source,'seek'): self.source.seek(0)
  return av.open(self.source, metadata_errors='replace')
 def _validate(self,f):
  if f.is_corrupt: raise ValueError('Corrupt decoded frame')
  shape=(f.width,f.height)
  if self.shape is not None and shape!=self.shape: raise ValueError('Variable resolution')
  self.shape=shape
 def _rgb(self,f):
  if self.rgb_mode=="x86-compatible" and f.format.name in ("yuv420p","yuvj420p") and f.width%2==0 and f.height%2==0 and int(f.colorspace) in (0,1,2,3,4,5,6,7,9):
   from .portable_rgb import convert_yuv420
   return convert_yuv420(f.to_ndarray(format=f.format.name),f.height,f.width,colorspace=int(f.colorspace),full_range=f.format.name.startswith("yuvj") or int(f.color_range)==2,implementation=self.rgb_implementation)
  return f.to_ndarray(format="rgb24")
 def __len__(self): return self.count
 def get_avg_fps(self): return self.fps
 def get_batch(self,indices):
  wanted=set(indices);selected={};count=0
  with self._open() as c:
   v=c.streams.video[0];v.thread_count=self.threads
   for i,f in enumerate(c.decode(v)):
    self._validate(f);count+=1
    if i in wanted:
     if int(f.color_primaries) in (0,3):f.color_primaries=2
     if int(f.color_trc) in (0,3):f.color_trc=2
     selected[i]=None if int(f.colorspace)==10 else self._rgb(f)
  if count!=self.count and self._count_hint_used:raise FrameCountHintMismatch('Container frame count differs from decoded count')
  if count!=self.count or wanted!=set(selected): raise ValueError('Decode count changed')
  if any(v is None for v in selected.values()):
   import tempfile,subprocess,imageio_ffmpeg,os
   # Only BT.2020 constant-luminance frames use zimg; other RGB conversions stay unchanged.
   missing=sorted(i for i,v in selected.items() if v is None)
   with tempfile.TemporaryDirectory(prefix='istock-bt2020cl-') as d:
    path=os.path.join(d,'input.mp4')
    if hasattr(self.source,'seek'):
     self.source.seek(0)
     with open(path,'wb') as out:out.write(self.source.read())
    else:path=os.fspath(self.source)
    select='+'.join('eq(n\\,%d)'%i for i in missing)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-xerror','-threads','1','-i',path,'-an','-vf',"select="+select+",zscale=matrix=gbr,format=gbrp,format=rgb24",'-fps_mode','passthrough','-threads','1','-f','rawvideo','-']
    result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    width,height=self.shape
    expected=len(missing)*width*height*3
    if len(result.stdout)!=expected:raise ValueError('zscale frame count/size mismatch')
    values=np.frombuffer(result.stdout,dtype=np.uint8).reshape(len(missing),height,width,3)
    selected.update(zip(missing,values))
  return Batch(np.stack([selected[i] for i in indices]))
