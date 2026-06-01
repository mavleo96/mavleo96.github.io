---
layout: page
title: Whisper Accent
description: Conditioning via adaptive layer normalization for accent-aware English speech recognition
img: assets/img/projects/whisper-accent/asr_thumbnail.png
importance: 1
category: "machine learning"
related_publications: false
---

Despite impressive multilingual performance, state-of-the-art ASR models like Whisper continue to exhibit higher word error rates (WER) on non-native and regionally diverse English accents. Training a separate adapter per accent is expensive and brittle: it demands sufficient labelled data for each accent, scales poorly, and discards shared structure across accents.

We present **Whisper-Accent**: an extension of pretrained Whisper that improves WER across 23 phonetically diverse English accents in a single model, with substantial gains on most, by conditioning the frozen decoder on per-accent learned embeddings via **Adaptive Layer Normalization (AdaLN)**. Only the AdaLN modulation weights, accent embeddings, and accent classifier are trained from scratch; the encoder and decoder backbone remain completely frozen. Whisper-Accent achieves **14.1% WER** (`whisper-accent-small.en`) and **13.4% WER** (`whisper-accent-medium.en`) compared to 17.6% and 17.5% for the respective Whisper baselines — absolute improvements of **3.5 and 4.1 percentage points**.

---

## Architecture

<div class="row justify-content-sm-center">
  <div class="col-sm-10 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/whisper-accent/architecture.png" title="Whisper-Accent Architecture" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Figure 1: Whisper-Accent architecture. The encoder predicts the accent label; the corresponding embedding conditions every frozen decoder LayerNorm via AdaLN.
</div>

**Accent Classifier**: The encoder produces hidden states at every layer. We learn scalar fusion weights over all $$L$$ encoder layers plus the input embedding, yielding a weighted-average representation of shape $$(T, D)$$. A linear projection reduces dimensionality, and multi-head attention pooling (MHAP) collapses the temporal axis via a learnable query vector; MHAP was found to outperform mean pooling empirically. The resulting fixed-length vector is passed to a linear classification head over $$A$$ accent classes.

**Accent Embeddings**: A lookup table of $$A$$ trainable vectors maps a predicted accent label to a conditioning vector $$e \in \mathbb{R}^{d/2}$$. Ground-truth labels are used during training; predicted labels from the classifier are used at inference, making the system fully self-contained.

**Adaptive Layer Normalization**: AdaLN was popularized in class-conditional diffusion transformers as a way to condition generation on a class embedding without modifying the attention or feed-forward weights — the same principle we adopt here for accent conditioning. Every LayerNorm in the Whisper decoder is replaced by an AdaLN module:

$$\text{AdaLN}(h, e) = \tilde{\gamma}(e) \odot \hat{h} + \tilde{\beta}(e)$$

where $$\hat{h} = \text{LayerNorm}(h)$$ is the normalized hidden state without scale or shift, and $$\tilde{\gamma}(e) = W_\gamma e + \gamma_0$$, $$\tilde{\beta}(e) = W_\beta e + \beta_0$$ are affine projections of the accent embedding. The projection weights $$W_\gamma, W_\beta$$ are zero-initialized and the biases are set to the pretrained LayerNorm parameters $$(\gamma_0, \beta_0)$$; so at initialization, AdaLN exactly reproduces the original Whisper LayerNorm behavior (following ControlNet).

---

## Two-Stage Training

Training is split into two stages because naive joint training is classifier-dominated: the randomly initialized classifier and zero-initialized AdaLN weights produce gradient norms differing by ~50x, leaving the AdaLN weights with negligible updates. Zero-initialization is required to preserve vanilla Whisper behavior at the start of training, making this imbalance unavoidable.

**Stage 1 — Accent Classifier**: The layer-fusion weights, projection, MHAP, and classification head are trained under pure accent cross-entropy ($$\lambda_\text{CE} = 0$$, $$\lambda_\text{accent} = 1$$) with class weighting to handle label imbalance, at a learning rate of `1e-3`.

**Stage 2 — Decoder AdaLN + Accent Embeddings**: Starting from the Stage 1 checkpoint, only the AdaLN modulation parameters and accent embedding table are unfrozen. Training uses pure ASR cross-entropy ($$\lambda_\text{CE} = 1$$, $$\lambda_\text{accent} = 0$$) conditioned on ground-truth accent labels, with learning rates of `5e-5` for AdaLN and `5e-4` for accent embeddings. Weight decay is disabled, consistent with zero-initialized weights.

---

## Results

All models are trained and evaluated on the [westbrook/English_Accent_DataSet](https://huggingface.co/datasets/westbrook/English_Accent_DataSet), a 79-hour speech corpus covering 23 English accents sourced from VCTK, EDACC, and VoxPopuli, with 50.4k training, 1.04k validation, and 1.62k test utterances. All results are on the test split.

### Comparison with Whisper Baselines

A single Whisper-Accent model outperforms both vanilla Whisper baselines and a stronger fine-tuned baseline (decoder LayerNorm fine-tuning). Vanilla Whisper WER remains flat across model sizes (17.5–17.7%), indicating that scaling alone does not improve accent robustness without explicit accent adaptation.

<div class="row justify-content-center">
  <div class="col-sm-8" markdown="1">

| Model                          |   Overall WER ↓    |
| :----------------------------- | :----------------: |
| _Whisper Baselines_            |                    |
| `whisper-small.en`             |       17.6%        |
| `whisper-medium.en`            |       17.5%        |
| `whisper-large-v3`             |       17.7%        |
| `whisper-large-v3-turbo`       |       20.1%        |
| _Decoder LayerNorm Fine-tuned_ |                    |
| `whisper-small.en`             |       17.2%        |
| `whisper-medium.en`            |       16.6%        |
| _Whisper-Accent (Ours)_        |                    |
| **`whisper-accent-small.en`**  | **14.1%** (↓3.5pp) |
| **`whisper-accent-medium.en`** | **13.4%** (↓4.1pp) |

  </div>
</div>
<div class="caption">
  Table 1: Overall WER on the English Accent Dataset test split. LayerNorm fine-tuned refers to fine-tuning decoder LayerNorm parameters without accent conditioning.
</div>

### Per-Accent WER and Accent Classification Accuracy

Improvements are observed across all 23 accent classes, with substantial gains on most — the magnitude varies from 18.6pp for Romanian to 0.2pp for Northern Irish. The classifier achieves 95.7% accuracy on `whisper-accent-medium.en` and 85.1% on `whisper-accent-small.en`. Native varieties reach near-perfect transcription (American: 1.2%, Canadian: 0.8%), while some accents remain challenging — Vietnamese: 32.3%, Indian English: 61.4%. Indian English is a notable outlier: WER improvement is negligible (↓0.9pp) despite sufficient training data and perfect classification accuracy — a pattern examined in the embedding analysis below.

<div class="row justify-content-center">
  <div class="col-sm-6" markdown="1">

| Accent         |     WER ↓     | Accent Acc. ↑ |  n  |
| :------------- | :-----------: | :-----------: | :-: |
| English        |  5.2% (↓0.3)  |     95.7%     | 442 |
| American       |  1.2% (↓0.4)  |     97.7%     | 263 |
| Scottish       |  6.9% (↓1.2)  |     94.9%     | 235 |
| Irish          |  9.7% (↓2.0)  |     97.4%     | 152 |
| Canadian       |  0.8% (↓1.4)  |    100.0%     | 90  |
| Northern Irish |  2.8% (↓0.2)  |     94.5%     | 73  |
| Indian         | 61.4% (↓0.9)  |    100.0%     | 51  |
| Spanish        | 14.8% (↓4.8)  |     95.7%     | 46  |
| Dutch          | 17.2% (↓18.3) |    100.0%     | 35  |
| Polish         | 14.8% (↓2.6)  |     96.8%     | 31  |
| Italian        |  8.6% (↓8.3)  |     86.2%     | 29  |
| German         | 18.1% (↓11.0) |     96.3%     | 27  |

  </div>
  <div class="col-sm-6" markdown="1">

| Accent      |      WER ↓       | Accent Acc. ↑ |    n     |
| :---------- | :--------------: | :-----------: | :------: |
| French      |   21.8% (↓3.2)   |     73.1%     |    26    |
| Romanian    |  14.3% (↓18.6)   |     91.3%     |    23    |
| Czech       |   10.1% (↓6.0)   |     94.7%     |    19    |
| Hungarian   |   9.7% (↓8.8)    |     83.3%     |    18    |
| Slovak      |   7.3% (↓9.5)    |     94.1%     |    17    |
| Vietnamese  |   32.3% (↓3.9)   |    100.0%     |    14    |
| Estonian    |   12.4% (↓6.5)   |    100.0%     |    13    |
| Finnish     |   8.6% (↓5.1)    |     81.8%     |    11    |
| Lithuanian  |   2.7% (↓2.7)    |    100.0%     |    2†    |
| Croatian    |  21.8% (↓10.9)   |    100.0%     |    2†    |
| Slovene     |   6.1% (↓9.1)    |     0.0%      |    1†    |
| **Overall** | **13.4%** (↓4.1) |   **95.7%**   | **1620** |

  </div>
</div>
<div class="caption">
  Table 2: Per-accent WER and classifier accuracy for whisper-accent-medium.en on the test split. n = number of test samples. † Results with n≤2 test samples are reported for completeness only. Per-accent improvements on moderate-n accents should be interpreted with caution in the absence of formal significance testing.
</div>

### Ablation: Ground-Truth vs. Predicted vs. Random Accent Labels

<div class="row justify-content-center">
  <div class="col-sm-8" markdown="1">

| Conditioning               | WER (small) | WER (medium) |
| :------------------------- | :---------: | :----------: |
| Ground-truth accent label  |    14.2%    |    13.4%     |
| **Predicted accent label** |  **14.1%**  |  **13.4%**   |
| Random accent label        |    16.6%    |    15.1%     |

  </div>
</div>
<div class="caption">
  Table 3: WER under different accent conditioning strategies at evaluation time.
</div>

Random conditioning still outperforms vanilla Whisper (16.6% / 15.1% vs. 17.6% / 17.5%): even a random accent embedding activates the AdaLN pathway and yields a measurable WER improvement, whereas vanilla Whisper has no conditioning pathway at all. The result is less surprising than it appears — with minimum pairwise embedding similarity of ~0.35 (visible in Figure 2), a random draw is a noisy blend over a relatively tight cluster rather than an arbitrary perturbation, which bounds the downside. The gap between random and predicted conditioning (2.5 pp / 1.7 pp) quantifies the net contribution of accurate accent classification. The near-zero gap between predicted and ground-truth (0.0–0.1 pp) suggests that classifier errors contribute negligibly to overall WER.

---

## Accent Embedding Analysis

<div class="row justify-content-sm-center">
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/whisper-accent/embedding_cosine_similarity.png" title="Cosine similarity heatmap of accent embeddings" class="img-fluid rounded z-depth-1" %}
  </div>
  <div class="col-sm-6 mt-3 mt-md-0">
    {% include figure.liquid loading="eager" path="assets/img/projects/whisper-accent/embedding_umap_projection.png" title="UMAP projection of accent embeddings" class="img-fluid rounded z-depth-1" %}
  </div>
</div>
<div class="caption">
  Figure 2: Left — cosine similarity matrix of the 23 learned accent embeddings (whisper-accent-medium.en). Right — UMAP projection of the same embeddings.
</div>

The cosine similarity heatmap reveals three broad regions: native English varieties, Western European L2 accents (Dutch, German, French, Polish), and Baltic/Slavic/Finnic accents (Hungarian, Czech, Slovak, Estonian, Lithuanian, Croatian, Slovene). Spanish and Italian occupy an intermediate region closer to native English. The groupings do not cleanly map to phonological families — Romanian clusters with native English varieties in the UMAP projection, yet its cosine similarity to the English embedding is 0.512 and it improves substantially (↓18.6pp from a 32.9% vanilla WER baseline). The UMAP grouping alone is therefore an incomplete picture of embedding geometry.

The clearest failure is Indian English: despite 1,351 training samples and perfect classification accuracy, its learned embedding collapses into the native English cluster (cosine similarity 0.875 — higher than any other non-native accent), yet vanilla WER is the highest in the dataset (62.3%) with near-zero improvement (↓0.9pp). Dutch (1,148 samples) and Romanian (553 samples) have comparable or fewer training samples yet their embeddings land substantially further from English, improving by 18.3pp and 18.6pp respectively — ruling out data volume as the explanation. Why the Stage 2 gradient failed to differentiate the Indian English embedding remains an open question; one hypothesis is the high internal phonological diversity of Indian English speakers, which may prevent the embedding from converging to a coherent direction, though this cannot be tested without speaker-level phonological metadata. Vietnamese shows a similar but less severe pattern: cosine similarity 0.678 to English, improvement of 3.9pp.

## Conclusion

Whisper-Accent demonstrates that a single frozen Whisper backbone can be conditioned across 23 English accents via AdaLN decoder modulation, with only 9.1% and 8.2% of total parameters trained for the medium and small models respectively. The approach is effective across most accents, with the primary failure mode being embedding collapse — where the learned accent embedding fails to differentiate from the native English cluster, conditioning provides little benefit regardless of sample count. What drives this collapse remains unresolved by the current experiments. A natural direction for future work is encoder-side accent adaptation, targeting the acoustic representations where accent information originates rather than conditioning the decoder after the fact. Code and pretrained checkpoints are available at [github.com/mavleo96/whisper-accent](https://github.com/mavleo96/whisper-accent).
