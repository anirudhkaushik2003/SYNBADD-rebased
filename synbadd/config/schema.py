"""Typed configuration schema for SYNBADD.

All config values are defined here as dataclasses. After loading from YAML,
the config is frozen (immutable) to prevent accidental mutation during training.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GeneratorConfig:
    type: str = "SVFRegressorDiffeoGenerator"
    nb_unet_features: List[List[int]] = field(
        default_factory=lambda: [[16, 32, 32, 32, 32], [32, 32, 32, 32, 32, 32, 16, 16]]
    )
    int_steps: int = 10
    int_downsize: int = 1
    using_daft: bool = True
    using_unet: bool = True
    use_deformation_field: bool = True


@dataclass
class TemporalConfig:
    type: str = "GatedCrossAttention"
    num_heads: int = 8
    num_layers: int = 2
    max_seq_len: int = 5
    use_positional_embeddings: bool = True
    use_bidirectional_embeddings: bool = False


@dataclass
class DiscriminatorConfig:
    enabled: bool = False
    type: str = "DAFTDiscriminator"
    channel_number: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 64])


@dataclass
class VAEConfig:
    type: str = "VAE3D_UNetSkip"
    frozen: bool = True


@dataclass
class ModelConfig:
    img_height: int = 128
    latent_dim: int = 512
    c_dim: int = 3
    in_channels: int = 3
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    vae: VAEConfig = field(default_factory=VAEConfig)


@dataclass
class LossConfig:
    ssim: float = 0.2
    lddmm: float = 0.5
    jacobian: float = 0.1
    jac_dist: float = 0.05
    smooth: float = 0.05
    disc: float = 0.05
    epe_x: float = 0.0
    epe_y: float = 0.0
    epe_z: float = 0.0
    angular: float = 0.0


@dataclass
class TrainingConfig:
    n_epochs: int = 200
    batch_size: int = 6
    effective_batch_size: int = 6
    lr: float = 5e-5
    betas: Tuple[float, float] = (0.5, 0.999)
    patience: int = 5
    early_stopping_patience: int = 7
    min_lr: float = 1e-6
    aug_p: float = 0.0
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class DataConfig:
    dataset: str = "ADNI"
    fold: int = 1
    target_timepoint: str = "all"
    timepoints: List[str] = field(default_factory=lambda: ["sc", "m06", "m12", "m18", "m24", "m36"])
    svf_type: str = "baseline"
    warp_baseline: bool = True
    use_past_data: bool = True
    use_future_data: bool = True
    use_val_dataset: bool = True


@dataclass
class BaselinesConfig:
    noder: str = ""
    remind: str = ""


@dataclass
class PathsConfig:
    base_dir: str = "/scratch/anirudhkaushik"
    csv_file: str = ""
    svf_metadata: str = ""
    svf_dir: str = ""
    svf_baseline_dir: str = ""
    vae_checkpoint: str = ""
    vxm_checkpoint: str = ""
    checkpoint_dir: str = ""
    output_dir: str = ""
    masks_dir: str = ""
    brain_masks_dir: str = ""
    seg_dir: str = ""
    baseline_seg_dir: str = ""
    baselines: BaselinesConfig = field(default_factory=BaselinesConfig)

    def resolve_defaults(self):
        """Fill empty paths with defaults derived from base_dir."""
        defaults = {
            "csv_file": f"{self.base_dir}/ADNI1_Complete_DataSplits.csv",
            "svf_metadata": f"{self.base_dir}/svf_metadata.csv",
            "svf_dir": f"{self.base_dir}/displacement_fields_processed",
            "svf_baseline_dir": f"{self.base_dir}/svf_outputs",
            "vae_checkpoint": f"{self.base_dir}/SYNBADD/VAE_10_Aug_per_sample/best_model.pth",
            "vxm_checkpoint": f"{self.base_dir}/vxm_checkpoints/best_vxm_model_64.pth",
            "checkpoint_dir": f"{self.base_dir}/SYNBADD",
            "output_dir": f"{self.base_dir}/SYNBADD_outputs",
            "masks_dir": f"{self.base_dir}/ADNI_Resized_Registered_Affine",
            "brain_masks_dir": f"{self.base_dir}/ADNI_Resized_Registered_brain_masks",
            "seg_dir": f"{self.base_dir}/ADNI_Resized_Registered_segs",
            "baseline_seg_dir": f"{self.base_dir}/ADNI_Resized_Registered_baseline_segs",
        }
        for attr, default in defaults.items():
            if not getattr(self, attr):
                object.__setattr__(self, attr, default)

        if not self.baselines.noder:
            object.__setattr__(self.baselines, "noder", f"{self.base_dir}/NODER_result")
        if not self.baselines.remind:
            object.__setattr__(self.baselines, "remind", f"{self.base_dir}/remind_results")


@dataclass
class SystemConfig:
    n_gpu: int = 2
    n_cpu: int = 3
    seed: int = 42
    debug: bool = False


@dataclass
class WandbConfig:
    enabled: bool = True
    project: str = "SYNBADD_Experiments"
    entity: Optional[str] = None


@dataclass
class EvalConfig:
    baselines: List[str] = field(default_factory=lambda: ["noder", "remind"])
    roi_regions: Dict[str, List[int]] = field(
        default_factory=lambda: {
            "hippocampus": [17, 53],
            "amygdala": [18, 54],
            "entorhinal_cortex": [1006, 2006],
            "parahippocampal": [1016, 2016],
            "fusiform_gyrus": [1007, 2007],
            "inferior_temporal_gyrus": [1009, 2009],
            "middle_temporal_gyrus": [1015, 2015],
            "third_ventricle": [14],
        }
    )


@dataclass
class SynbaddConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Runtime fields (set by CLI/pipeline, not in YAML)
    run_name: str = ""
    project_name: str = ""
    specials: str = ""

    def __post_init__(self):
        self.paths.resolve_defaults()
