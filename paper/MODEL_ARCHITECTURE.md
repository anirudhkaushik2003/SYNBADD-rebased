# SynBADD Main-Model Architecture

The configured main model is **CrossAttentionTransformer + 3D U-Net + DAFT**, using both past and future baseline-referenced stationary velocity fields (SVFs).

## Architecture diagram

```mermaid
flowchart TB
    subgraph Inputs["Longitudinal input"]
        direction TB
        B["Baseline MRI<br/>1 × 128³"]
        S["Five follow-up SVF slots<br/>5 × 3 × 128³<br/>target and missing slots zero-filled"]
        C["Condition vector<br/>diagnosis, age, target visit<br/>3 scalars"]
        Q["Target slot index<br/>0…4"]
    end

    subgraph FrozenVAE["Frozen 3D SVF VAE — applied independently to each slot"]
        direction TB
        VE["Encoder<br/>3 → 32 → 64 → 128 → 256 → 512 → 512"]
        MU["Posterior mean μ<br/>512-D per slot"]
        SK["Five skip hierarchies<br/>temporal mean at each scale"]
    end

    subgraph Temporal["Proposed target-aware cross-attention transformer"]
        direction TB
        POS["Learned positional embeddings<br/>5 × 512"]
        SELF["2-layer Transformer encoder<br/>d = 512, 8 heads<br/>FFN = 2048, dropout = 0.1"]
        TQ["Learned target-time embedding<br/>qₚ ∈ R⁵¹²"]
        CA["Target-conditioned dot-product attention<br/>aᵢ = softmax(qₚᵀhᵢ)"]
        Z["Weighted temporal latent<br/>ẑₚ = Σᵢ aᵢhᵢ<br/>512-D"]
    end

    subgraph Decode["Frozen VAE decoder"]
        direction TB
        VD["FC → 512 × 4³<br/>five upsampling stages"]
        CSVF["Coarse predicted SVF<br/>3 × 128³"]
    end

    subgraph Refine["DAFT-conditioned 3D U-Net"]
        direction TB
        UE["Encoder<br/>3 → 16 → 32 → 32 → 32 → 32<br/>128³ → 4³"]
        UD["Decoder with spatial skips<br/>32 → 32 → 32 → 32 → 32"]
        DAFT["Three full-resolution DAFT residual blocks<br/>35 → 32 → 16 → 16<br/>conditioned by diagnosis, age, target visit"]
        FLOW["3D convolution<br/>16 → 3<br/>refined stationary velocity field"]
    end

    subgraph Synthesis["Diffeomorphic synthesis"]
        direction TB
        INT["Scaling-and-squaring integration<br/>10 steps, full 128³ resolution"]
        STN["Spatial transformer<br/>trilinear sampling"]
        OUT["Synthetic target MRI<br/>1 × 128³"]
    end

    S --> VE
    VE --> MU
    VE --> SK
    MU --> POS --> SELF
    Q --> TQ
    SELF --> CA
    TQ --> CA
    CA --> Z
    Z --> VD
    SK --> VD
    VD --> CSVF
    CSVF --> UE --> UD --> DAFT --> FLOW
    C --> DAFT
    FLOW --> INT --> STN
    B --> STN
    STN --> OUT
```

## Configured parameters

| Component | Main setting |
|---|---|
| Input image/SVF resolution | `128 × 128 × 128` |
| Longitudinal SVF slots | 5: `m06, m12, m18, m24, m36` |
| SVF channels | 3 |
| VAE latent dimension | 512 |
| VAE encoder widths | `32, 64, 128, 256, 512, 512` |
| Temporal model | `CrossAttentionTransformer` |
| Transformer layers | 2 |
| Self-attention heads | 8 |
| Transformer FFN | 2048, inherited PyTorch default |
| Maximum sequence length | 5 |
| Positional embeddings | Learned |
| U-Net encoder | `16, 32, 32, 32, 32` |
| U-Net decoder | `32, 32, 32, 32, 32, 32, 16, 16` |
| DAFT condition dimension | 3 |
| Integration | 10 scaling-and-squaring steps |
| Integration resolution | Full resolution, `int_downsize = 1` |
| Discriminator | Disabled |
| VAE | Pretrained and frozen |

## Implementation findings

### 1. Cross-attention formulation

The longitudinal tokens first pass through an ordinary multi-head self-attention encoder. The learned target-timepoint embedding then attends to the encoded tokens using:

$$
a_i = \operatorname{softmax}_i(q_p^\top h_i),
\qquad
\hat z_p = \sum_i a_i h_i.
$$

The self-attention encoder is eight-headed, but the final target-conditioned cross-attention is a single dot-product pooling operation rather than `nn.MultiheadAttention`.

### 2. Missing slots are not attention-masked

The dataset creates a five-slot sequence and zero-fills:

- the target timepoint;
- unavailable visits;
- visits excluded by past/future ablations.

The generated `mask` is not supplied to the VAE or transformer. A zero SVF can produce a nonzero VAE latent because it passes through learned convolutions and batch-normalisation. The transformer can therefore attend to nominally missing slots.

### 3. VAE skip features are averaged without masking

At each VAE resolution, the implementation calculates a simple mean across all five temporal slots. This includes zero-filled target and missing slots after they have passed through the VAE encoder. This behaviour may contribute to the near-invariance observed in the missing-input experiments.

### 4. VAE depth and normalisation

The implementation has six encoder stages:

$$
3 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256
\rightarrow 512 \rightarrow 512.
$$

It uses `BatchNorm3d`, not instance normalisation. The current paper description should be updated accordingly.

### 5. DAFT placement

DAFT is not applied at every U-Net layer. It is applied in three full-resolution residual blocks after the encoder–decoder path. Each block combines globally pooled image features with:

$$
[\text{diagnosis},\ \text{age},\ \text{target visit index}]
$$

and predicts channel-wise scale and shift parameters.

### 6. Condition encoding

The condition variables are passed as raw scalars:

- diagnosis: `CN = 0`, `MCI = 1`, `AD = 2`;
- subject age;
- target visit index.

They are not one-hot encoded or normalised inside the model implementation.

### 7. Baseline source image

Despite being named `latest_scan`, the source image is always loaded from the screening visit (`sc`). The final spatial transformer therefore warps the screening baseline, consistent with the baseline-referenced formulation.

### 8. Constructed but unused parameter paths

Two parameterised paths are currently constructed but unused in the main forward pass:

- `CrossAttentionTransformer.attention_fc` is instantiated but never called.
- `SVFRegressorDiffeoGenerator.project`, a roughly 6.3-million-parameter latent-to-volume projection, is instantiated but never called because the transformer output is decoded by the VAE before entering the generator.

## Implementation sources

- [`configs/experiments/cross_attn_DAFT_PF.yaml`](../configs/experiments/cross_attn_DAFT_PF.yaml)
- [`synbadd/models/temporal/cross_attention.py`](../synbadd/models/temporal/cross_attention.py)
- [`synbadd/models/vae.py`](../synbadd/models/vae.py)
- [`synbadd/models/generator.py`](../synbadd/models/generator.py)
- [`synbadd/models/daft_blocks.py`](../synbadd/models/daft_blocks.py)
- [`synbadd/models/spatial_ops.py`](../synbadd/models/spatial_ops.py)
- [`synbadd/engine/trainer.py`](../synbadd/engine/trainer.py)
- [`synbadd/data/base_dataset.py`](../synbadd/data/base_dataset.py)
