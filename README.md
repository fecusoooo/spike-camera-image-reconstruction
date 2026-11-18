# Spike-Camera Image Reconstruction

A course project studying spike-camera image reconstruction, reproducing the BSF method and comparing it with TFI, TFP, and SSML on real and public spike data.

## Overview

This project investigates image reconstruction from spike-camera data. We used a **Spike M1K40-H2-Gen3** camera to record real spike streams, examined the data decoding and preprocessing required by the camera's `.dat` format, and evaluated reconstruction methods.

The main method studied is **BSF** (*Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations*, CVPR 2024). The report compares BSF with three other methods:

- **TFI** — a conventional, non-learning reconstruction method.
- **TFP** — a conventional, non-learning reconstruction method.
- **SSML** — a self-supervised method; the full version with attention and mutual-learning components was used in the reported experiments.

We also made an exploratory modification to BSF's MODF feature-fusion stage using a physical-prior gate.

## Repository Structure

```text
.
├── BSF/                    # BSF-related code and resources
├── card_data_result/       # Experimental result files
├── our_data_result/        # Results from the team's real recordings
├── rot_data_result/        # Experimental result files
├── 第25组_实验报告.pdf       # Original project report (Chinese)
└── README.md
```

## Report and References

- **Project report (Chinese):** [`第25组_实验报告.pdf`](./第25组_实验报告.pdf)
- **BSF paper:** Rui Zhao et al., “Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations,” CVPR 2024.
- **SpikeCV:** Open-source spike-camera vision framework used for the TFI, TFP, and SSML scripts described in the report.

For the exact experimental setup, preprocessing details, additional figures, and discussion, see the project report.

---

*This README summarizes the team's course-project report and repository contents. It distinguishes the reported experimental observations from the exploratory Safe PG-MODF modification and its stated limitations.*
