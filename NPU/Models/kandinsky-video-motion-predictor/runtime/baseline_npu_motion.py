"""NPU adapter: original model, normalization and per-clip head semantics."""
import torch
from motion_only import MotionScorer

class NpuMotionScorer(MotionScorer):
    @torch.inference_mode()
    def infer_batch(self, batch):
        m, nf = self.model, self.model.num_frames
        frames = torch.cat(batch).to(self.device, non_blocking=True).float() / 255.0
        segs = [len(b)//nf for b in batch]
        with torch.autocast('npu',dtype=torch.float16,enabled=self.fp16):
            feat=m.forward_encoder(frames.reshape(-1,nf,*frames.shape[1:]),max_size=self.max_segments)
            out=[]
            for f in feat.split(segs):
                score=m.forward_head(f,torch.tensor([len(f)],device=self.device))
                if m.use_sigmoid:
                    score=torch.sigmoid(score)*m.targets_norm
                out.append(score[0].float().cpu().tolist())
        return [dict(motion_camera=c,motion_object=o,motion_dynamics=d) for c,o,d in out]
