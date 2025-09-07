from typing import Any
import torch
from dataclasses import dataclass, asdict
import numpy as np
import argparse
import os
from tqdm import tqdm
import wandb
from cs336_basics import data_loader, optimizer, transformer, config_schema
from cs336_basics.config_schema import load_config


def logsumexp(logits):
    max_logit = logits.max(dim=-1, keepdim=True).values
    x = logits - max_logit
    return x.exp().sum(dim=-1).log() + max_logit.squeeze(-1)


def cross_entropy(logits, targets):
    # logits: b, t, v
    # targets: b, t
    lse = logsumexp(logits)

    v = logits.shape[-1]
    one_hot = (targets[..., None] == torch.arange(
        v, device=logits.device)).float()  # b, t, v
    target_logit = (one_hot * logits).sum(dim=-1)
    return - (target_logit - lse).mean()


def perplexity(logits, targets):
    return cross_entropy(logits, targets).exp()


@dataclass
class TrainState:
    model_params: dict[str, Any]
    optimizer: dict[str, Any]
    iteration: int
    model_config: config_schema.ModelConfig


def save_checkpoint(model, optimizer, iteration, model_config, out):
    train_state = TrainState(
        model_params=model.state_dict(),
        optimizer=optimizer.state_dict(),
        iteration=iteration,
        model_config=model_config
    )
    torch.save(asdict(train_state), out)


def load_checkpoint(src, model, optimizer):
    train_state = TrainState(**torch.load(src))

    model_params = train_state.model_params
    model.load_state_dict(model_params)
    optimizer.load_state_dict(train_state.optimizer)
    return train_state.iteration


def load_model_from_checkpoint(checkpoint_path, compile=False, device=None):
    train_state = TrainState(**torch.load(checkpoint_path))
    model = transformer.TransformerLM(**train_state.model_config)
    if compile:
        if device == "mps":
            model = torch.compile(model, backend="aot_eager")
        else:
            model = torch.compile(model)
    if device is not None:
        model = model.to(device)

    # Backward compatibility: handle old checkpoints with torch.compile prefix
    model_params = train_state.model_params
    # if any(key.startswith('_orig_mod.') for key in model_params.keys()):
    #     model_params = {key.replace('_orig_mod.', ''): value
    #                     for key, value in model_params.items()}

    model.load_state_dict(model_params)
    return model


def init_wandb(config):
    """Initialize wandb logging if enabled."""
    if not config.wandb.enabled:
        return None

    # Prepare wandb config
    wandb_config = {
        # Model config
        "model": config.model.__dict__,
        # Optimizer config
        "optimizer": config.optimizer.__dict__,
        # Data config
        # Exclude device for cleaner logs
        "data": {k: v for k, v in config.data.__dict__.items() if k != "device"},
        # Training config
        "training": config.training.__dict__,
    }

    # Initialize wandb
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        name=config.wandb.name,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config=wandb_config
    )

    return wandb


def main():
    parser = argparse.ArgumentParser(description="Train GPT")
    parser.add_argument("--config", type=str,
                        default="cs336_basics/config.yaml",
                        help="Path to config")
    args = parser.parse_args()

    # Load configuration using the new schema
    config = load_config(args.config)
    data_config = config.data
    model_config = config.model
    opt_config = config.optimizer
    training_config = config.training

    # Initialize wandb logging
    wandb_run = init_wandb(config)

    # Set random seed
    torch.manual_seed(training_config.seed)
    np.random.seed(training_config.seed)

    # Create checkpoint directory with wandb run ID if available
    if wandb_run is not None:
        experiment_checkpoint_dir = os.path.join(
            training_config.checkpoint_dir, wandb.run.id)
        print(f"Using wandb run ID for checkpoints: {wandb.run.id}")
    else:
        experiment_checkpoint_dir = training_config.checkpoint_dir

    os.makedirs(experiment_checkpoint_dir, exist_ok=True)

    # Load datasets
    train_dataset = np.memmap(data_config.train_data_path, dtype=np.uint16)
    val_dataset = np.memmap(data_config.val_data_path, dtype=np.uint16)

    # Initialize model
    original_model = transformer.TransformerLM(
        vocab_size=model_config.vocab_size,
        context_length=model_config.context_length,
        d_model=model_config.d_model,
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        d_ff=model_config.d_ff,
        rope_theta=model_config.rope_theta
    ).to(data_config.device)

    # Keep reference to original model for checkpointing
    model = original_model
    try:
        if data_config.device == "mps":
            model = torch.compile(original_model, backend="aot_eager")
        else:
            model = torch.compile(original_model)
    except:
        print("Failed to compile model")

    if opt_config.name == "adamw":
        opt = optimizer.AdamW(model.parameters(),
                              opt_config.lr, opt_config.betas, opt_config.eps)
    else:
        raise ValueError(f"Invalid optimizer: {opt_config.name}")

    try:
        for step in tqdm(range(training_config.num_steps), desc="Training"):
            model.train()
            x, y = data_loader.get_batch(
                train_dataset, data_config.batch_size, model_config.context_length, data_config.device)

            logits = model(x)
            loss = cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()

            if opt_config.gradient_clip_norm is not None:
                optimizer.clip_gradient(model.parameters(),
                                        opt_config.gradient_clip_norm)
            activation_norm = logits.square().sum().sqrt()
            grad_norm = sum([p.grad.square().sum()
                             for p in model.parameters()]).sqrt()
            weight_norm = sum([p.square().sum()
                              for p in model.parameters()]).sqrt()

            # set lr schedule
            lr = optimizer.get_lr_cosine_schedule(
                step, opt_config.lr_max, opt_config.lr_min, opt_config.warmup_steps, opt_config.annealing_steps)
            for p in opt.param_groups:
                p['lr'] = lr

            opt.step()

            # Prepare metrics dictionary for this step
            metrics_to_log = {}

            if step % training_config.eval_interval == 0:
                model.eval()
                x_val, y_val = data_loader.get_batch(
                    val_dataset, data_config.batch_size, model_config.context_length, data_config.device)
                logits_val = model(x_val)
                val_loss = cross_entropy(logits_val, y_val)
                val_perplexity = perplexity(logits_val, y_val)

                # Add validation metrics to the log dictionary
                metrics_to_log.update({
                    "val/loss": val_loss.item(),
                    "val/perplexity": val_perplexity.item(),
                })

            if step % training_config.save_interval == 0:
                checkpoint_path = os.path.join(
                    experiment_checkpoint_dir, f"checkpoint_{step}.pt")
                save_checkpoint(model, opt, step,
                                model_config, checkpoint_path)

                # Log model checkpoint to wandb if enabled
                if wandb_run is not None and config.wandb.log_model:
                    wandb.save(checkpoint_path)

            if step % training_config.log_interval == 0:
                train_perplexity = perplexity(logits, y)

                # Add training metrics to the log dictionary
                metrics_to_log.update({
                    "train/loss": loss.item(),
                    "train/perplexity": train_perplexity.item(),
                    "train/learning_rate": lr,
                    "train/grad_norm": grad_norm.item(),
                    "train/weight_norm": weight_norm.item(),
                    "train/activation_norm": activation_norm.item(),
                    "train/token_seen": data_config.batch_size * model_config.context_length * step,
                })

            # Log all metrics for this step in a single wandb.log() call
            if wandb_run is not None and metrics_to_log:
                wandb.log(metrics_to_log, step=step)

    finally:
        # Ensure wandb run is properly finished
        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
