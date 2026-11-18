# Spike-Camera Image Reconstruction

A course project studying spike-camera image reconstruction, reproducing the BSF method and comparing it with TFI, TFP, and SSML on real and public spike data.

## Overview

This project investigates image reconstruction from spike-camera data. The team used a **Spike M1K40-H2-Gen3** camera to record real spike streams, examined the data decoding and preprocessing required by the camera's `.dat` format, and evaluated reconstruction methods.

The main method studied is **BSF** (*Boosting Spike Camera Image Reconstruction from a Perspective of Dealing with Spike Fluctuations*, CVPR 2024). The report compares BSF with three other methods:

- **TFI** — a conventional, non-learning reconstruction method.
- **TFP** — a conventional, non-learning reconstruction method.
- **SSML** — a self-supervised method; the full version with attention and mutual-learning components was used in the reported experiments.

The repository also records an exploratory modification to BSF's MODF feature-fusion stage using a physical-prior gate.

## Data Acquisition

### Camera and scene

The real-data experiment used a Spike M1K40-H2-Gen3 camera with a spatial resolution of **1000 × 1000** and a sampling rate of **20,000 Hz**. A continuously illuminated scene was recorded using a photography light placed to the side of the subject.

The main test scene used a horizontally positioned, rotating electric fan. The fan blades carried the letters “C-A-M-E-R-A”; a mechanical clock was placed in the background as a static reference. This setup was intended to examine motion deblurring while preserving background details.

### Raw data

The camera recording software saved the one-second recording as 50 sequential `.dat` files. The report describes concatenating these files into a single `combined.dat` file of approximately 2.0 GB, containing 160,602 time steps.

### Decoding and preprocessing

The report documents several details needed to decode the camera's binary stream:

1. Skip the 64-byte frame header.
2. Read each row as 1024 packed bits, although the physical image width is 1000 pixels.
3. Unpack the little-endian bit stream.
4. Remove the invalid padded columns to recover a `(T, 1000, 1000)` spike array.
5. Correct quadrant alignment and apply the required image flip.

The report notes that directly interpreting the binary data as a `1000 × 1000` array produces incorrect results. The preprocessing implementation is documented in the report appendix and referenced there as `infer_our_data.py`.

## Reconstruction Methods and Evaluation

### Temporal alignment

The BSF model takes **61 consecutive spike frames** as input and reconstructs the center frame. To produce a sequence, the experiment uses a sliding window with a stride of **3**, discarding the first and last 30 frames. The report applies the same temporal alignment to TFI, TFP, and SSML so that the output frames correspond to the same input windows.

### Method-specific settings

- **TFI, TFP, and SSML** are run using corresponding scripts based on the open-source SpikeCV framework.
- For **TFP**, the reported `half_win_length` is **20** (41 frames in total). The report explains that a shorter window produced visibly noisier results and was less comparable in overall image quality to the other methods.
- The **full SSML** version, including its attention module and mutual-learning mechanism, is used.

### Quality measures

The project evaluates reconstructed images using the no-reference image-quality measures **BRISQUE**, **PIQE**, and **NIQE**, and also reports the **Laplacian variance** as an indicator of image sharpness.

The report cautions that no-reference image-quality metrics can be misleading for spike reconstruction: smoother images may receive better BRISQUE/PIQE/NIQE scores even when they lose details, while sharper reconstructions may retain visible high-frequency noise.

## Results and Discussion

### Real rotating-fan recording

In the reported real-camera sequence, BSF reconstructs the rapidly rotating fan blades with sharp edges and legible “C-A-M-E-R-A” lettering, showing strong motion-deblurring performance. The report also observes visible grain-like high-frequency noise in dark background regions.

### Quantitative results

The following values are reproduced from Table 2 of the project report. BRISQUE, PIQE, and NIQE are lower-is-better; Laplacian variance is higher-is-sharper.

| Method | BRISQUE ↓ | PIQE ↓ | NIQE ↓ | Laplacian variance ↑ |
| --- | ---: | ---: | ---: | ---: |
| TFI | 47.31 ± 1.75 | 54.40 ± 0.63 | 8.92 ± 0.15 | 1539.91 ± 137.60 |
| TFP | 57.73 ± 0.50 | 62.54 ± 0.36 | 9.27 ± 0.10 | 1500.16 ± 4.63 |
| SSML | 2.17 ± 0.44 | 24.07 ± 0.35 | 4.21 ± 0.07 | 295.40 ± 1.08 |
| BSF | 22.19 ± 0.66 | 45.51 ± 0.36 | 4.00 ± 0.07 | 924.44 ± 3.61 |

These metrics do not provide a single consistent ranking. In particular, the report discusses how the smoother SSML outputs obtain low no-reference metric scores while having much lower Laplacian variance. The authors therefore recommend complementing no-reference metrics with downstream-task evaluation, such as detection, segmentation, feature matching, or optical-flow/depth estimation.

### Challenging lighting conditions

The report also discusses two difficult real-world cases:

- **Strong exposure with fast motion:** high photon flux can saturate spike output, while rapid displacement makes temporal alignment more difficult.
- **Low light:** sparse spikes and low signal-to-noise ratio make it difficult to recover fine details; timing fluctuations and dark-count noise can become more visible in the reconstruction.

## Exploratory Modification: Safe PG-MODF

The report explores retaining the original MODF convolution structure while adding a physical-prior gate and a safe feature-fusion step. The gate is based on consistency measures between low- and high-order DSFT features:

- a difference map measuring disagreement between low- and high-order features;
- a variance map measuring the internal stability of high-order features;
- an exponential gate that reduces the contribution of high-order residual features when these measures indicate instability.

The report describes this as a lightweight, no-retraining structural modification with manually selected hyperparameters. It largely preserves the original BSF performance, but does **not** report a significant improvement in PSNR or LPIPS. Since the team's real recordings do not include corresponding ground-truth images, the report identifies supervised evaluation on those recordings as a limitation and proposes further work on learnable gating and self-supervised training.

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
