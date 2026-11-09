# Occlusion-Aware Fingerprints (OAF)

Anonymous code release for the CVPR 2027 submission *"Occlusion-Aware Fingerprints for Consistent Cross-Camera Person Identity"*.

OAF is a **training-free, test-time** plug-in for any video or image ReID model.
1. A frozen pose estimator (YOLO11m-pose) checks every 4th frame of a clip.
2. A frame is kept if a person is found and at least 50% of the 17 body keypoints are visible (confidence ≥ 0.5).
3. Frames that were not checked follow the nearest checked frame.
4. If fewer than 8 frames pass, the clip is left unchanged.
5. The unchanged ReID model builds the clip fingerprint from the kept frames only.

No labels, no training, and no change to the ReID model are needed.

## Repository layout

```
oaf/select.py        reference implementation of the frame selection (OAFSelector, keep_mask)
oaf/protocol.py      consistent-identity evaluation (clustering, calibrated cut, 50 tests)
notebooks/           every experiment in the paper, as Google Colab notebooks (run in order)
notebooks/exploratory/   early alternatives we tried before OAF (not in the paper)
```

Quick use of the selector:

```python
from oaf import OAFSelector
sel = OAFSelector()              # paper defaults
keep = sel(frames)               # frames: list of images of one clip -> boolean mask
fingerprint = reid_model(frames[keep])
```

`python -m oaf.select` runs a self-test of the selection rule.

## Notebooks and the paper

| Notebook | What it does | Paper |
|---|---|---|
| `00_build_MEVID-SC` | Builds MEVID-SC: one outfit per person (the outfit seen on the most cameras), using labels only; verifies it and writes the dataset statistics | Sec. 3, Tab. 1 |
| `01_CSCI_features_MEVID` | Runs the released CSCI MEVID checkpoint with the authors' code; reproduces 56.7 mAP / 79.7 Rank-1 | Tab. 6, 7 |
| `02_CSCI_baseline_MEVID-SC` | CSCI fingerprints restricted to MEVID-SC (identical clips), consistent-ID protocol | Tab. 4 |
| `03_OSNet_baseline_MEVID-SC` | OSNet x1_0 (MSMT17) clip fingerprints on MEVID-SC (cell 1 builds them) | Tab. 4 (OSNet) |
| `04_diagnosis_factors_MEVID-SC` | Eleven-factor study of identity splits (person fixed effects, permutation-corrected R², bootstrap) | Sec. 4, Tab. 3, Fig. 2 |
| `05_OAF_CSCI_MEVID-SC` | Pose checks, filtered clips, CSCI rerun on the kept frames | Tab. 4 |
| `06_OAF_controls_MEVID-SC` | Random and feature-distance removal of the same number of frames | Tab. 4 |
| `07_OAF_three_models_MEVID-SC` | CSCI, FastReID SBS R101-IBN, OSNet × 4 variants; cell 6 gives the final numbers | Tab. 4 |
| `08_OAF_three_models_full_MEVID` | Same study on the full official MEVID test (all outfits); cell 8 gives the outfit breakdown | Tab. 6, 7, Sec. 6 |
| `09_validation_iLIDS-VID_PRID-2011` | Zero-shot no-harm check on hand-cropped data | Tab. 1, Sec. 6 |
| `10_validation_MARS` | Same study on MARS (prepared; data access pending, not in the paper) | — |
| `11_ablation_threshold_CSCI` | CSCI with the visibility threshold at 0 (person found), 0.25 and 0.75 | Tab. 8 |
| `12_ablation_OSNet_cost_qualitative` | OSNet threshold ablation, run-time cost, qualitative figures | Tab. 8, Sec. 6, Fig. 1, 4 |
| `13_alternatives_same_protocol_MEVID-SC` | Camera z-norm, camera S-norm and k-reciprocal re-ranking, alone and combined with OAF, under the identical 50-test protocol (CPU only) | Sec. 6 "Alternatives we tried", Supp. Tab. S1, S2 |
| `14_occlusion_split_rate_before_after` | Split rate by occlusion level (none / some / heavy / rarely visible), before and after OAF and for both controls (CPU only) | Tab. 5 |

The `exploratory/` notebooks (camera normalisation, camera adaptation, risk-controlled linking, a learned verifier and a clip-level quality gate) are early alternatives we tried before OAF, under earlier settings; they are not in the paper and are included for transparency. The alternatives reported in the paper were re-run under the paper's exact protocol in notebook 13.

## Data and weights

- **MEVID**: obtain from the official MEVID release (Davila et al., WACV 2023). The notebooks expect `mevid-v1-bbox-test.tgz` and `mevid-v1-annotation-data.zip` (plus the train archive for notebook 00).
- **CSCI**: code and MEVID checkpoint (`ez_eva02_vid_hybrid_extra_best.pth`) from the authors' public repository `ppriyank/ICCV-CSCI-Person-ReID`. Their code is used unchanged, apart from a hook that saves the distance matrix and features, and progress printing.
- **FastReID** SBS R101-IBN (MSMT17) from the FastReID GitHub release. **OSNet** x1_0 (MSMT17) from torchreid.
- **YOLO11m-pose** from Ultralytics (downloaded automatically).
- iLIDS-VID, PRID-2011 and MARS come from their official sources.

## Running

The notebooks run on Google Colab (an A100 or L4 for the GPU steps).
- **Drive layout.** They mount Google Drive and look for data under a `Data/` folder: `Data/MEVID/`, `Data/MEVID-SameClothes/`, with results in `Data/MEVID-results/`. Adjust the `SD` / search paths in cell 1 if your layout differs.
- **Saving and resuming.** Every long step saves to Drive in chunks and resumes after a restart, and every long loop shows a progress bar.
- **Restarts.** The results cells are standalone and reload everything from Drive.

CSCI runs may print a numpy `in1d` traceback **after** the features are saved; it comes from CSCI's optional extra evaluation protocol under numpy 2 and does not affect the saved features or any result.

All settings were fixed before any test result and are written to `SETTINGS_frozen.json` by each notebook:
- pose check every 4 frames, input 320, detection confidence 0.25;
- keypoint confidence 0.5, visibility threshold 0.5, minimum 8 kept frames;
- precision target 0.90, 25 person splits, seed 2027;
- re-ranking (notebook 13): k1 = 20, k2 = 6, lambda = 0.3 (standard settings).

## Main results (mean over 50 tests)

| | MEVID-SC IDF1 | Full MEVID IDF1 | Full MEVID mAP |
|---|---|---|---|
| CSCI | 81.66 → **85.77** | 66.23 → **71.03** | 56.66 → **59.67** |
| FastReID | 41.72 → **46.68** | 30.91 → **36.00** | 16.12 → **20.10** |
| OSNet | 40.97 → **48.73** | 30.91 → **36.05** | 17.62 → **20.22** |

## Alternatives under the same protocol (MEVID-SC IDF1, mean over 50 tests; notebook 13)

| | CSCI | FastReID | OSNet |
|---|---|---|---|
| All frames | 81.66 | 41.72 | 40.97 |
| Camera z-norm | 67.46 | 37.88 | 49.51 |
| Camera S-norm | 42.48 | 43.43 | 41.64 |
| k-reciprocal re-ranking | 84.89 | 44.35 | 42.75 |
| **OAF** | **85.77** | **46.68** | **48.73** |
| OAF + re-ranking | 86.62 | 49.40 | 45.94 |

Camera normalisation often misses the 90% pair-precision target (down to 81%); see supplementary Table S1 for all metrics and Table S2 for the paired tests.

## License

MIT (see `LICENSE`). Third-party code and weights keep their own licenses.
