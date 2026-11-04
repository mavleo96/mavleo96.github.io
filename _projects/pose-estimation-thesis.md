---
layout: page
title: Novel Object Pose Estimation
description: MS thesis on training-free novel object 6DoF pose estimation by fusing diffusion and DINOv2 features
img: assets/img/projects/pose_estimation_thesis.png
github: https://github.com/mavleo96/pose-estimation
importance: 1
category: "machine learning"
related_publications: false
---

Novel object pose estimation requires estimating the full 6DoF pose of unseen objects using only a CAD model, without object-specific training. Discriminative features such as DINOv2 are effective but lack spatial smoothness and struggle on textureless objects. Diffusion models produce complementary spatially smooth representations, but their utility in the synthetic-to-real setting has not been studied.

This thesis proposes a **training-free pipeline** that fuses diffusion and DINOv2 features into 3D point descriptors and recovers pose via **Kabsch+RANSAC**. Evaluated on all seven core BOP datasets, the fused pipeline outperforms DINOv2 alone with a mean gain of **+6.6 Average Recall (AR)**, with gains of up to **+8.5 AR** on TUD-L and YCB-V and **+6.0 AR** on LM-O. It also outperforms DZOP (an RGB-only method using diffusion features) on LM-O and YCB-V. Ablations show that low-noise features (t ≤ 50) and all four decoder layers are preferable for object-level matching.

Master of Science thesis in Computer Science, Stony Brook University (May 2026), advised by Prof. Zhaozheng Yin.

- **Thesis (PDF):** [Diffusion Feature Fusion for Novel Object Pose Estimation]({{ '/assets/pdf/ms_thesis_vijayabharathi_updated.pdf' | relative_url }})
- **Code:** [github.com/mavleo96/pose-estimation](https://github.com/mavleo96/pose-estimation)
