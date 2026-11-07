"""Occlusion-aware fingerprints (OAF): training-free frame selection.

A frozen pose estimator checks every `check_every`-th frame of a clip (and the
last frame). A checked frame passes if a person is detected and at least
`visible_thr` of the 17 COCO keypoints have confidence >= `keypoint_conf`.
Every frame takes the decision of its nearest checked frame. If fewer than
`min_kept` frames pass, the clip is left unchanged (all frames are used).

The ReID model is untouched: feed it only the kept frames.

Default settings are the ones used in the paper (fixed before any result).
"""
from __future__ import annotations

import numpy as np

DEFAULTS = dict(check_every=4, pose_model='yolo11m-pose.pt', imgsz=320,
                det_conf=0.25, keypoint_conf=0.5, visible_thr=0.5, min_kept=8)


def check_indices(n_frames: int, check_every: int = 4) -> np.ndarray:
    """Indices of the frames that are checked: 0, k, 2k, ... and the last frame."""
    c = list(range(0, n_frames, check_every))
    if c[-1] != n_frames - 1:
        c.append(n_frames - 1)
    return np.asarray(c)


def keep_mask(n_frames: int, checked: np.ndarray, found: np.ndarray,
              vis: np.ndarray, visible_thr: float = 0.5,
              min_kept: int = 8) -> np.ndarray:
    """Boolean keep-mask over all frames of one clip.

    checked: indices of checked frames; found: 1 if a person was detected;
    vis: share of the 17 keypoints with confidence >= keypoint_conf.
    """
    checked = np.asarray(checked)
    fr = np.arange(n_frames)
    near = np.abs(fr[:, None] - checked[None, :]).argmin(1)
    ok = (np.asarray(found) == 1) & (np.asarray(vis) >= visible_thr)
    keep = ok[near]
    if keep.sum() < min_kept:          # too few good frames: clip unchanged
        return np.ones(n_frames, bool)
    return keep


class OAFSelector:
    """Runs the pose check on a list of frames (BGR or RGB numpy images)."""

    def __init__(self, device=None, half=True, **kw):
        from ultralytics import YOLO      # pip install ultralytics
        self.cfg = {**DEFAULTS, **kw}
        self.model = YOLO(self.cfg['pose_model'])
        self.device, self.half = device, half

    def check(self, images):
        """Return (found, vis) for each image."""
        c = self.cfg
        found = np.zeros(len(images), np.int8)
        vis = np.zeros(len(images), np.float32)
        res = self.model.predict(list(images), imgsz=c['imgsz'], conf=c['det_conf'],
                                 verbose=False, half=self.half, device=self.device)
        for i, (im, r) in enumerate(zip(images, res)):
            if len(r.boxes):
                H, W = im.shape[:2]
                xy = r.boxes.xyxy.cpu().numpy()
                cf = r.boxes.conf.cpu().numpy()
                area = (xy[:, 2] - xy[:, 0]) * (xy[:, 3] - xy[:, 1]) / (H * W)
                m = int(np.argmax(area * cf))   # main person: largest area x confidence
                kp = r.keypoints.data.cpu().numpy()[m]
                found[i] = 1
                vis[i] = (kp[:, 2] >= c['keypoint_conf']).mean()
        return found, vis

    def __call__(self, frames):
        """frames: list of images of one clip. Returns the keep-mask over all frames."""
        c = self.cfg
        idx = check_indices(len(frames), c['check_every'])
        found, vis = self.check([frames[i] for i in idx])
        return keep_mask(len(frames), idx, found, vis, c['visible_thr'], c['min_kept'])


if __name__ == '__main__':
    # self-test of the selection rule (no GPU, no model needed)
    idx = check_indices(20)
    assert list(idx) == [0, 4, 8, 12, 16, 19]
    m = keep_mask(20, idx, found=[1, 1, 0, 0, 1, 1], vis=[.9, .6, 0, 0, .3, .8], min_kept=4)
    assert m[:7].all() and not m[7:15].any()
    assert keep_mask(20, idx, found=[0] * 6, vis=[0] * 6).all()   # fallback
    print('ok')
