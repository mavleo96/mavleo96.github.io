---
layout: page
title: Whisper Accent
description: Conditioning via adaptive layer normalization for accent-aware English speech recognition
img: assets/img/projects/whisper-accent/asr_thumbnail.png
importance: 1
category: "machine learning"
related_publications: false
---

Despite impressive multilingual performance, state-of-the-art ASR models like Whisper continue to exhibit elevated word error rates (WER) on non-native and regionally diverse English accents. Training a separate adapter per accent is expensive and brittle: it demands sufficient labelled data for each accent, scales poorly, and discards the shared phonological structure that spans accent families.

We present **Whisper-Accent**: an extension of pretrained Whisper that handles 23 phonetically diverse English accents in a single model by conditioning the frozen decoder on per-accent learned embeddings via **Adaptive Layer Normalization (AdaLN)**. Only the AdaLN modulation weights, accent embeddings, and accent classifier are trained from scratch; the encoder and decoder backbone remain completely frozen. Whisper-Accent achieves **14.1% WER** (`whisper-accent-small.en`) and **13.4% WER** (`whisper-accent-medium.en`) compared to 17.6% and 17.5% for the respective Whisper baselines — absolute improvements of **3.5 and 4.1 percentage points**.

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

**Accent Classifier.** The encoder produces hidden states at every layer. We learn scalar fusion weights over all $$L$$ encoder layers plus the input embedding, yielding a weighted-average representation of shape $$(T, D)$$. A linear projection reduces dimensionality, and multi-head attention pooling (MHA-pool) collapses the temporal axis via a learnable query vector. The resulting fixed-length vector is passed to a linear classification head over $$A$$ accent classes.

**Accent Embeddings.** A lookup table of $$A$$ trainable vectors maps a predicted accent label to a conditioning vector $$e \in \mathbb{R}^{d/2}$$. Ground-truth labels are used during training; predicted labels from the classifier are used at inference, making the system fully self-contained.

**Adaptive Layer Normalization.** AdaLN was popularized in class-conditional diffusion transformers as a way to condition generation on a class embedding without modifying the attention or feed-forward weights — the same principle we adopt here for accent conditioning. Every LayerNorm in the Whisper decoder is replaced by an AdaLN module:

$$\text{AdaLN}(h, e) = \tilde{\gamma}(e) \odot \hat{h} + \tilde{\beta}(e)$$

where $$\hat{h} = \text{LayerNorm}(h)$$ is the normalized hidden state without scale or shift, and $$\tilde{\gamma}(e) = W_\gamma e + \gamma_0$$, $$\tilde{\beta}(e) = W_\beta e + \beta_0$$ are affine projections of the accent embedding. The projection weights $$W_\gamma, W_\beta$$ are zero-initialized and the biases are set to the pretrained LayerNorm parameters $$(\gamma_0, \beta_0)$$; so at initialization, AdaLN exactly reproduces the original Whisper LayerNorm behavior (following ControlNet). Because the backbone is entirely frozen, the model preserves Whisper's original generalization capability for accents outside the training distribution.

---

## Two-Stage Training

Training is split into two stages to manage the large difference in gradient norms between the randomly initialized accent classifier and the zero-initialized AdaLN weights.

**Stage 1 — Accent Classifier.** Only the layer-fusion weights, projection, MHA pooling, and classification head are trained under pure accent cross-entropy ($$\lambda_\text{CE} = 0$$, $$\lambda_\text{accent} = 1$$) with class weighting to handle label imbalance. Learning rate: `1e-3`.

**Stage 2 — Decoder AdaLN + Accent Embeddings.** From the Stage 1 checkpoint, only the AdaLN modulation parameters and accent embedding table are unfrozen. Training uses pure ASR cross-entropy ($$\lambda_\text{CE} = 1$$, $$\lambda_\text{accent} = 0$$) conditioned on ground-truth accent labels. Learning rate: `5e-5` for AdaLN; `5e-4` for accent embeddings. Weight decay is disabled, consistent with zero-initialized weights.

---

## Results

All models are trained and evaluated on the [westbrook/English_Accent_DataSet](https://huggingface.co/datasets/westbrook/English_Accent_DataSet), a 79-hour speech corpus covering 23 English accents sourced from VCTK, EDACC, and VoxPopuli, with 50.4k training, 1.04k validation, and 1.62k test utterances. All results are on the test split.

### Comparison with Whisper Baselines

A single Whisper-Accent model outperforms both vanilla Whisper baselines and a stronger fine-tuned baseline (decoder LayerNorm fine-tuning) — and even the much larger `whisper-large-v3` — demonstrating that accent conditioning is a more effective lever than raw model scale or naive adaptation.

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

Improvements are observed across all 23 accent classes. The classifier achieves 95.7% accuracy on `whisper-accent-medium.en` and 85.1% on `whisper-accent-small.en`. Native varieties reach near-perfect transcription (American: 1.2%, Canadian: 0.8%), while phonologically distant accents remain challenging — Vietnamese: 32.3%, Indian English: 61.4%, the latter likely compounded by a small test set (n=51).

<div class="row justify-content-center">
  <div class="col-sm-6" markdown="1">

| Accent         |      WER ↓      | Accent Acc. ↑ |  n  |
| :------------- | :-------------: | :-----------: | :-: |
| English        |  5.2% (↓0.3pp)  |     95.7%     | 442 |
| American       |  1.2% (↓0.4pp)  |     97.7%     | 263 |
| Scottish       |  6.9% (↓1.2pp)  |     94.9%     | 235 |
| Irish          |  9.7% (↓2.0pp)  |     97.4%     | 152 |
| Canadian       |  0.8% (↓1.4pp)  |    100.0%     | 90  |
| Northern Irish |  2.8% (↓0.2pp)  |     94.5%     | 73  |
| Indian         | 61.4% (↓0.9pp)  |    100.0%     | 51  |
| Spanish        | 14.8% (↓4.8pp)  |     95.7%     | 46  |
| Dutch          | 17.2% (↓18.3pp) |    100.0%     | 35  |
| Polish         | 14.8% (↓2.6pp)  |     96.8%     | 31  |
| Italian        |  8.6% (↓8.3pp)  |     86.2%     | 29  |
| German         | 18.1% (↓11.0pp) |     96.3%     | 27  |

  </div>
  <div class="col-sm-6" markdown="1">

| Accent      |       WER ↓        | Accent Acc. ↑ |    n     |
| :---------- | :----------------: | :-----------: | :------: |
| French      |   21.8% (↓3.2pp)   |     73.1%     |    26    |
| Romanian    |  14.3% (↓18.6pp)   |     91.3%     |    23    |
| Czech       |   10.1% (↓6.0pp)   |     94.7%     |    19    |
| Hungarian   |   9.7% (↓8.8pp)    |     83.3%     |    18    |
| Slovak      |   7.3% (↓9.5pp)    |     94.1%     |    17    |
| Vietnamese  |   32.3% (↓3.9pp)   |    100.0%     |    14    |
| Estonian    |   12.4% (↓6.5pp)   |    100.0%     |    13    |
| Finnish     |   8.6% (↓5.1pp)    |     81.8%     |    11    |
| Lithuanian  |   2.7% (↓2.7pp)    |    100.0%     |    2     |
| Croatian    |  21.8% (↓10.9pp)   |    100.0%     |    2     |
| Slovene     |   6.1% (↓9.1pp)    |     0.0%      |    1     |
| **Overall** | **13.4%** (↓4.1pp) |   **95.7%**   | **1620** |

  </div>
</div>
<div class="caption">
  Table 2: Per-accent WER and classifier accuracy for whisper-accent-medium.en on the test split. n = number of test samples.
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

Random conditioning still outperforms vanilla Whisper (16.6% / 15.1% vs. 17.6% / 17.5%), which is expected: with minimum pairwise embedding similarity of ~0.4, a randomly drawn embedding acts as a noisy weighted average over the embedding cluster rather than a true null. The gap between random and predicted conditioning (2.5 pp / 1.7 pp) therefore quantifies the net contribution of accurate accent classification. The near-zero gap between predicted and ground-truth (0.0–0.1 pp) confirms that the Stage 1 classifier effectively matches oracle performance.

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

The cosine similarity heatmap reveals three blocks: native English varieties form the tightest cluster reflecting genuine phonological coherence. Western European accents (Dutch, German, French, Polish) and Baltic/Slavic/Finnic accents (Hungarian, Czech, Slovak, etc.) cluster internally with nearly identical similarity, despite phonological distance. This counterintuitive grouping emerges from training data distribution—accents with similar sample sufficiency converge in embedding space regardless of typology.

The UMAP projection corroborates the three-region structure but reveals critical collapse patterns. Native English varieties occupy a coherent region. Western European and Baltic/Slavic accents project into spatially proximate clusters, confirming their embedding equivalence. Notable failures: Indian and Vietnamese project into the native English region despite phonological distinctness, visible as spatial collapse in UMAP and high cosine similarity to native English. Romanian, by contrast, embeds apart from native English and separates in UMAP, anchored by phonologically similar peers in the Romance/Germanic L2 group. Low-resource accents without peer representation collapse into the majority cluster; those with peers maintain structure.

---

## Conclusion

Whisper-Accent shows that phonetically diverse accents can be handled within a single model by conditioning a frozen Whisper decoder on per-accent embeddings via AdaLN, eliminating the need for per-accent adapters. The approach works well for data-rich accents, but the embedding analysis highlights a key limitation: low-resource accents fail from phonological isolation, not data scarcity. Indian and Vietnamese achieve perfect classification yet minimal WER improvement, while Romanian with similar sample count achieves substantial improvement due to peer anchoring from Romance/Germanic L2 accents. Addressing this will require contrastive embedding objectives encouraging phonological grouping, combined with adding low-resource accents with shared L1 typology for peer representation. Code and pretrained checkpoints are available at [github.com/mavleo96/whisper-accent](https://github.com/mavleo96/whisper-accent).
