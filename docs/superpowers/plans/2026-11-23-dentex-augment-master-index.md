# DENTEX Tooth-Swap Augmentation Pipeline: Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Master Goal:** Implement the complete FDI-constrained Tooth-Swap and Lesion Copy-Paste augmentation subsystem for DENTEX dental panoramic radiographs (`dentex_bench.augment`), featuring bit-exact reproducible recipe execution, multi-mode blending, automated QC seam evaluation, and end-to-end integration with detector benchmarks.

**Reference Documents:**
- Technical Specification: [docs/superpowers/specs/DENTEX_AUGMENT_SPEC.md](file:///d:/Research/Dentex/docs/superpowers/specs/DENTEX_AUGMENT_SPEC.md)
- Scientific Rationale & Literature Review: [docs/superpowers/specs/Tăng cường dữ liệu bằng ghép răng và tổn thương cho phát hiện bệnh lý trên X-quang toàn cảnh DENTEX_ tổng quan, phản biện và pipeline thực nghiệm.md](file:///d:/Research/Dentex/docs/superpowers/specs/Tăng%20cường%20dữ%20liệu%20bằng%20ghép%20răng%20và%20tổn%20thương%20cho%20phát%20hiện%20bệnh%20lý%20trên%20X-quang%20toàn%20cảnh%20DENTEX_%20tổng%20quan,%20phản%20biện%20và%20pipeline%20thực%20nghiệm.md)

---

## Plan Decomposition & Execution Roadmap

Due to the breadth of the system (data extraction, geometry, image processing, state persistence, exporters, quality control, ablation suites), the implementation is decomposed into 3 bite-sized, independently testable phases:

| Phase | Plan Document | Milestones Covered | Key Deliverable |
|---|---|---|---|
| **Phase 1** | [2026-11-23-dentex-augment-p1-core-bank-placement.md](file:///d:/Research/Dentex/docs/superpowers/plans/2026-11-23-dentex-augment-p1-core-bank-placement.md) | M0, M1, M2 | Core utilities (`rng`, `fdi`, `config`, `io_utils`), data inspection (`inspect_dentex_json.py`), cached `ToothBank` builder, geometry engine (`geometry.py`), and anatomical placement engine (`placement.py`) with full rejection constraints. |
| **Phase 2** | [2026-11-23-dentex-augment-p2-blending-recipe-state-offline.md](file:///d:/Research/Dentex/docs/superpowers/plans/2026-11-23-dentex-augment-p2-blending-recipe-state-offline.md) | M3, M4 | Multi-mode blending (`blending.py`), label update synchronization (`labels.py`), pure deterministic recipe execution (`recipe.py`), state management (`state.py`), COCO/YOLO exporters, offline runner (`scripts/augment_offline.py`), and bit-exact replay tool (`scripts/replay_augment.py`). |
| **Phase 3** | [2026-11-23-dentex-augment-p3-qc-experiments-online-extras.md](file:///d:/Research/Dentex/docs/superpowers/plans/2026-11-23-dentex-augment-p3-qc-experiments-online-extras.md) | M5, M6, M7 | QC seam score analysis (`qc.py`, `scripts/qc_report.py`), E0-E9 presets and ablation orchestrator (`scripts/run_ablation.py`), detector training integration (`scripts/train.py`), online dataset wrapper (`SwapAugmentDataset`), RFS sampling, high-res patch cropping, hard-negative pasting, and classical mask segmentation (Otsu/GrabCut/Active Contour) benchmark. |

---

## Global Invariants (RFC 2119)

All tasks across all phases strictly adhere to the project invariants:
- **INV-1:** Bank is built **exclusively from the train split** (80/20, `seed=42`). Code asserts zero validation `image_id` leakage.
- **INV-2:** Validation and test sets are **never augmented**.
- **INV-3:** No vertical flip anywhere. Full-image rotation $\le \pm 5^\circ$.
- **INV-4:** Source tooth is placed only into the same tooth number (1–8) and jaw (upper/lower). Left-right quadrant swap requires horizontal patch mirroring.
- **INV-5:** Impacted teeth are placed exclusively into tooth position 8.
- **INV-6:** When replacing a target tooth: delete all annotations of target tooth, add all annotations of source tooth.
- **INV-7:** Saved images must be in lossless PNG format. No JPEG.
- **INV-8:** Executing a recipe is a pure function and must never call unseeded RNG.
