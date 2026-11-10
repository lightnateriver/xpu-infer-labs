import sys,os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
THIRD_PARTY=Path(__file__).resolve().parent/"third_party"
CKPT=Path(os.environ["QC_CKPT_DIR"]) if os.environ.get("QC_CKPT_DIR") else None
UPSTREAM=Path(os.environ.get("MOTION_UPSTREAM_DIR", str(THIRD_PARTY/"kandinsky-video-tools")))
def _video_reader(path, **kw):
    import decord

    return decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=2, **kw)

def _np(x):
    """decord batch -> np.ndarray (DOVER switches decord's global bridge to torch on import)."""
    return x.numpy() if isinstance(x, torch.Tensor) else x.asnumpy()

class MotionScorer:
    """kandinsky-video-tools VideoMotionPredictor: camera / object / dynamics motion in [0, 3]."""

    name = "motion"
    FRAME_SIZE = 224

    def __init__(self, device=None, load_model=True, fps=4.0, max_frames=240, fp16=True, max_segments=64):
        self.fp16 = fp16
        self.fps = fps
        self.max_frames = max_frames
        self.max_segments = max_segments
        if not load_model:
            return
        if CKPT is None:
            raise ValueError("Set QC_CKPT_DIR to the checkpoint root")
        sys.path.insert(0, str(UPSTREAM))
        from models import VideoMotionPredictor

        self.model = VideoMotionPredictor.from_pretrained(str(CKPT / "ai-forever/kandinsky-video-motion-predictor"))
        self.model = self.model.to(device).eval()
        assert self.model.frame_size == self.FRAME_SIZE
        self.device = device

    def prep(self, row, vr=None):
        # Same sampling/resizing as VideoMotionPredictor.preprocess_data, but only decodes the sampled frames.
        import imageio.v3 as iio

        vr = vr or _video_reader(row["clip_path"])
        total = len(vr)
        duration = iio.immeta(row["clip_path"], plugin="pyav")["duration"]  # official: same call
        num = min(self.fps * int(duration), self.max_frames)
        num = int((num // 16 + 1) * 16)
        idx = torch.linspace(0, total - 1, num).clamp(0, total - 1).long().tolist()
        uniq = sorted(set(idx))
        fr = torch.from_numpy(_np(vr.get_batch(uniq))).permute(0, 3, 1, 2)
        small = torch.cat([F.interpolate(c.float() / 255.0, size=(self.FRAME_SIZE,) * 2, mode="bilinear")
                           for c in fr.split(32)])
        pos = {f: i for i, f in enumerate(uniq)}
        return (small[[pos[i] for i in idx]] * 255).round_().to(torch.uint8)  # 4x less IPC, <=0.5/255 error

    @torch.inference_mode()
    def infer_batch(self, batch):
        # The transformer head has no padding mask, so batching whole videos would change the scores of
        # shorter clips. Batch only the VideoMAE backbone (the expensive part), run the head per clip.
        m, nf = self.model, self.model.num_frames
        frames = torch.cat(batch).to(self.device, non_blocking=True).float() / 255.0
        segs = [len(b) // nf for b in batch]
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.fp16):
            feat = m.forward_encoder(frames.reshape(-1, nf, *frames.shape[1:]), max_size=self.max_segments)
            out = []
            for f in feat.split(segs):
                o = m.forward_head(f, torch.tensor([len(f)], device=self.device))
                if m.use_sigmoid:
                    o = torch.sigmoid(o) * m.targets_norm
                out.append(o[0].float().cpu().tolist())
        return [{"motion_camera": c, "motion_object": o, "motion_dynamics": d} for c, o, d in out]
