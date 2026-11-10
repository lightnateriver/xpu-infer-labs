"""Validated optional NPU motion optimizations. Original paths remain defaults."""
from contextlib import nullcontext
import math
import torch
from motion_only import MotionScorer
from baseline_npu_motion import NpuMotionScorer

DEFAULT_CONFIG = {"workers":8,"decoder_threads":2,"order":"fifo","reuse_metadata":False,
                  "inplace_normalize":False,"head_fast":False,"persistent_amp":False}
SUPPORTED = {'reuse_metadata': True, 'inplace_normalize': False, 'head_fast': False, 'persistent_amp': False, 'decoder_threads4': True, 'lpt': True}

def validate_config(config=None):
    supplied=dict(config or {})
    unknown=set(supplied)-set(DEFAULT_CONFIG)
    if unknown:raise ValueError(f"Unknown options: {sorted(unknown)}")
    c=dict(DEFAULT_CONFIG,**supplied)
    for name in ["workers","decoder_threads"]:
        if isinstance(c[name],bool) or not isinstance(c[name],int) or c[name]<1:
            raise ValueError(f"{name} must be a positive integer")
    if c["decoder_threads"] not in ([2,4] if SUPPORTED["decoder_threads4"] else [2]):
        raise ValueError("Decoder thread count was not retained by this release")
    if c["order"] not in (["fifo","lpt"] if SUPPORTED["lpt"] else ["fifo"]):
        raise ValueError("Unsupported scheduling order")
    for key in ["reuse_metadata","inplace_normalize","head_fast","persistent_amp"]:
        if not isinstance(c[key],bool):raise ValueError(f"{key} must be boolean")
        if c[key] and not SUPPORTED[key]:raise ValueError(f"Rejected optimization: {key}")
    return c

def order_inputs(items,config):
    if config["order"]=="lpt":
        for item in items:
            if any(not isinstance(item.get(k),(int,float)) or not math.isfinite(item[k]) or item[k]<=0
                   for k in ["width","height","num_frames"]):
                raise ValueError("LPT requires positive width, height and num_frames metadata")
        return sorted(items,key=lambda r:r["width"]*r["height"]*r["num_frames"],reverse=True)
    return list(items)

def inference_context(config):
    return nullcontext()

class Preprocessor:
    def __init__(self):
        self.original=MotionScorer(load_model=False,max_segments=16)

    def __call__(self,item,config):
        reader=None
        if config.get("decoder_threads",2)!=2:
            import decord
            reader=decord.VideoReader(str(item["path"]),ctx=decord.cpu(0),num_threads=config["decoder_threads"])
        reuse=config.get("reuse_metadata",False)
        if not reuse:

            return self.original.prep({"clip_path":item["path"]},vr=reader)
        import imageio.v3 as iio
        import torch.nn.functional as F
        from motion_only import _video_reader,_np
        reader=reader or _video_reader(item["path"])
        total=len(reader)
        duration=item.get("motion_duration",item.get("duration_s")) if reuse else None
        if duration is None:duration=iio.immeta(item["path"],plugin="pyav")["duration"]
        if isinstance(duration,bool):raise ValueError("Invalid container duration")
        duration=float(duration)
        if not math.isfinite(duration) or duration<=0:raise ValueError("Invalid container duration")
        num=min(self.original.fps*int(duration),self.original.max_frames)
        num=int((num//16+1)*16)
        indices=torch.linspace(0,total-1,num).clamp(0,total-1).long().tolist()
        unique=sorted(set(indices))
        frames=torch.from_numpy(_np(reader.get_batch(unique))).permute(0,3,1,2)
        parts=[]
        for chunk in frames.split(32):
            x=chunk.float()
            x=x/255.0
            parts.append(F.interpolate(x,size=(self.original.FRAME_SIZE,)*2,mode="bilinear"))
            del x
        small=torch.cat(parts)
        del parts
        positions={f:i for i,f in enumerate(unique)}
        return (small[[positions[i] for i in indices]]*255).round_().to(torch.uint8)

OptimizedMotionScorer = NpuMotionScorer
