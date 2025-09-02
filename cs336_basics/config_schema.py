from dataclasses import dataclass, field
from typing import Optional, Tuple, Any, Dict
import yaml
import os
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 1600
    num_layers: int = 48
    num_heads: int = 25
    # If None, will be computed as 8/3 * d_model rounded to nearest 64
    d_ff: Optional[int] = None
    rope_theta: float = 10000.0

    def __post_init__(self):
        """Validate and set derived parameters."""
        if self.d_ff is None:
            self.d_ff = round(int(8 / 3 * self.d_model) / 64) * 64

        # Validate that d_model is divisible by num_heads
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer."""
    name: str = "adamw"  # "sgd" or "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    # Learning rate schedule
    use_lr_schedule: bool = True
    lr_max: Optional[float] = None  # If None, uses lr
    lr_min: float = 1e-5
    warmup_steps: int = 1000
    annealing_steps: int = 10000

    # Gradient clipping
    gradient_clip_norm: Optional[float] = 1.0

    def __post_init__(self):
        """Set derived parameters."""
        if self.lr_max is None:
            self.lr_max = self.lr


@dataclass
class DataConfig:
    """Configuration for data loading."""
    train_data_path: str = "data/train.bin"
    val_data_path: str = "data/val.bin"
    batch_size: int = 32
    device: str = "cuda" if hasattr(__import__('torch'), 'cuda') and __import__(
        'torch').cuda.is_available() else "cpu"


@dataclass
class WandbConfig:
    """Configuration for Weights & Biases logging."""
    enabled: bool = True
    project: str = "cs336-transformer"
    entity: Optional[str] = None  # Your wandb username/team
    name: Optional[str] = None  # Run name, if None will be auto-generated
    tags: Optional[list[str]] = None  # Tags for the run
    notes: Optional[str] = None  # Notes for the run
    log_model: bool = False  # Whether to log model checkpoints to wandb


@dataclass
class TrainingConfig:
    """Configuration for training."""
    num_steps: int = 10000
    eval_interval: int = 500
    save_interval: int = 1000
    log_interval: int = 100

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from_checkpoint: Optional[str] = None
    save_checkpoint: bool = True

    # Random seed
    seed: int = 42


@dataclass
class Config:
    """Main configuration class that combines all sub-configurations."""
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if config_dict is None:
            config_dict = {}

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from a dictionary."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        optimizer_config = OptimizerConfig(**config_dict.get('optimizer', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        wandb_config = WandbConfig(**config_dict.get('wandb', {}))

        return cls(
            model=model_config,
            optimizer=optimizer_config,
            data=data_config,
            training=training_config,
            wandb=wandb_config
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            'model': self.model.__dict__,
            'optimizer': self.optimizer.__dict__,
            'data': self.data.__dict__,
            'training': self.training.__dict__,
            'wandb': self.wandb.__dict__
        }

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to a YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)

        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def validate(self) -> None:
        """Validate the configuration."""
        # Model validation is done in __post_init__

        # Optimizer validation
        if self.optimizer.name not in ["sgd", "adamw"]:
            raise ValueError(f"Unsupported optimizer: {self.optimizer.name}")

        if self.optimizer.lr <= 0:
            raise ValueError(
                f"Learning rate must be positive: {self.optimizer.lr}")

        if self.optimizer.weight_decay < 0:
            raise ValueError(
                f"Weight decay must be non-negative: {self.optimizer.weight_decay}")

        # Data validation
        if self.data.batch_size <= 0:
            raise ValueError(
                f"Batch size must be positive: {self.data.batch_size}")

        # Training validation
        if self.training.num_steps <= 0:
            raise ValueError(
                f"Number of steps must be positive: {self.training.num_steps}")

        if self.training.eval_interval <= 0:
            raise ValueError(
                f"Eval interval must be positive: {self.training.eval_interval}")


def load_config(yaml_path: str) -> Config:
    """Convenience function to load and validate configuration from YAML."""
    config = Config.from_yaml(yaml_path)
    config.validate()
    return config


def create_default_config(yaml_path: str) -> Config:
    """Create and save a default configuration file."""
    config = Config()
    config.to_yaml(yaml_path)
    return config


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Config schema utilities")
    parser.add_argument("--create-default", type=str,
                        help="Create a default config file")
    parser.add_argument("--validate", type=str, help="Validate a config file")

    args = parser.parse_args()

    if args.create_default:
        config = create_default_config(args.create_default)
        print(f"Created default config at: {args.create_default}")

    if args.validate:
        try:
            config = load_config(args.validate)
            print(f"Config file {args.validate} is valid!")
            print("Configuration:")
            print(yaml.dump(config.to_dict(), default_flow_style=False, indent=2))
        except Exception as e:
            print(f"Config validation failed: {e}")
