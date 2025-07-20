"""Minimal training script for SmolLM.

Usage:
    python train_smol.py --model_name distilbert/distilgpt2 --dataset_name wikitext --max_length 32 --epochs 1 --batch_size 1
"""

import os
import re
import argparse
from datetime import timedelta

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.utilities import rank_zero_only
import datasets
import wandb
from datasets import load_from_disk
import lm_eval
from lm_eval import simple_evaluate
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling


from mtp._types import ModelHeadType
from mtp.mthf import MultiTokenHFConfig, MultiTokenHF

PRETRAINING_DS_CONFIG = {
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "name": "sample-10BT",
        "split": "train",
        # "streaming": True,
    },
    # small ds for testing
    "wikitext": {
        "path": "Salesforce/wikitext",
        "name": "wikitext-2-v1",
        "split": "test",
        "streaming": True,
    },
}


class LitLM(pl.LightningModule):
    def __init__(
        self, model_name, model_head: ModelHeadType = "stp", lr=5e-5, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Old
        # config = AutoConfig.from_pretrained(model_name)
        # # override config
        # for k, v in kwargs.items():
        #     setattr(config, k, v)
        # self.model = AutoModelForCausalLM.from_config(config)

        # New
        config = MultiTokenHFConfig(model_name=model_name, model_head=model_head)
        self.model = MultiTokenHF(config, **kwargs)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])


class LMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        dataset_name,
        batch_size,
        max_length,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length

    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.max_length:
            total_length = (total_length // self.max_length) * self.max_length
        # Split by chunks of block_size.
        result = {
            k: [
                t[i : i + self.max_length]
                for i in range(0, total_length, self.max_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def setup(self, stage=None):
        if self.dataset_name == "fineweb":
            self.dataset = load_from_disk("data/fineweb")
            # limit samples for testing
            # self.dataset = self.dataset.select(range(50000))
        else:
            self.dataset = datasets.load_dataset(
                **PRETRAINING_DS_CONFIG[self.dataset_name]
            )
            self.dataset = self.dataset.filter(
                lambda x: x["text"] and x["text"].strip() != ""
            )
            self.dataset = self.dataset.shuffle(seed=42)

            # Tokenize
            self.dataset = self.dataset.map(
                lambda x: self.tokenizer(x["text"]),
                remove_columns=["text"],
                batched=True,
            )

            # Group instead of padding/truncation
            self.dataset = self.dataset.map(
                lambda x: self.group_texts(x),
                batched=True,
            )

    def train_dataloader(self):
        collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=collator)


class HellaSwagEvalCallback(pl.Callback):
    def __init__(self, model_name, eval_every_n_batches=1, device=None):
        super().__init__()
        self.model_name = model_name
        self.eval_every_n_batches = eval_every_n_batches
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    @rank_zero_only
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (batch_idx + 1) % self.eval_every_n_batches == 0:
            print(
                f"\n[HellaSwagEvalCallback] Evaluating on HellaSwag at batch {batch_idx+1}..."
            )
            results = simple_evaluate(
                model=lm_eval.models.huggingface.HFLM(pretrained=pl_module.model),
                tasks=["hellaswag"],
                num_fewshot=0,
                batch_size=2,
                gen_kwargs={"max_new_tokens": 40},
                limit=100,
            )
            if (
                results
                and results.get("results")
                and results["results"].get("hellaswag")
            ):
                print(
                    f"[HellaSwagEvalCallback] HellaSwag results: {results['results']['hellaswag']}"
                )
                acc = results["results"]["hellaswag"].get("acc,none")
                acc_norm = results["results"]["hellaswag"].get("acc_norm,none")
                if acc_norm is not None:
                    log_dict = {
                        "eval/hellaswag_acc": acc,
                        "eval/hellaswag_acc_norm": acc_norm,
                        "batch": batch_idx + 1,
                    }
                    if (
                        hasattr(trainer.logger, "log_metrics")
                        and trainer.logger.log_metrics is not None
                    ):
                        trainer.logger.log_metrics(log_dict, step=batch_idx + 1)
            else:
                print("[HellaSwagEvalCallback] HellaSwag results not available.")


def get_econfig_name(args: argparse.Namespace):
    ignore_keys = ["disable_auto_resume"]
    parts = [f"{k[:1]}{v}" for k, v in args.__dict__.items() if k not in ignore_keys]
    # remove special characters
    parts = [re.sub(r"[^a-zA-Z0-9]", "", p) for p in parts]
    return "_".join(parts)


def lookup_ckpt(args: argparse.Namespace):
    ckpt_path = f"experiments/{get_econfig_name(args)}/last.ckpt"
    if not os.path.exists(ckpt_path):
        return None
    return ckpt_path


def lookup_wandb_run(args: argparse.Namespace):
    run_name = get_econfig_name(args)
    runs = wandb.Api(timeout=15).runs("mtl")
    matches = [r for r in runs if r.name == run_name]
    matches.sort(key=lambda x: x.created_at, reverse=True)
    if len(matches) == 0:
        return None
    return matches[0].id


# TODO:
# [x] add auto resume feature
# [ ] push to hub
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--model_head", type=str, default="stp")  # new
    p.add_argument("--dataset_name", type=str, default="fineweb")
    p.add_argument("--dataset_config", type=str, default="edu")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--disable_auto_resume", action="store_true")
    args = p.parse_args()

    # data
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    dm = LMDataModule(tokenizer, args.dataset_name, args.batch_size, args.max_length)

    # model
    model = LitLM(
        args.model_name,
        model_head=args.model_head,
        vocab_size=tokenizer.vocab_size,
    )

    # maybe auto resume
    resume_ckpt = lookup_ckpt(args)
    wandb_id = None
    if not (args.disable_auto_resume or resume_ckpt is None):
        print(f"[INFO] Resuming from checkpoint {resume_ckpt}.")
        resume_ckpt = lookup_ckpt(args)
        wandb_id = lookup_wandb_run(args)

    # trainer + callbacks
    eval_callback = HellaSwagEvalCallback(args.model_name, eval_every_n_batches=1000)
    wandb_logger = WandbLogger(
        project="mtl",
        name=get_econfig_name(args),
        id=wandb_id,
        resume="allow",
    )
    ckpt_best_callback = ModelCheckpoint(
        dirpath=f"experiments/{get_econfig_name(args)}",
        filename="best",
        monitor="eval/hellaswag_acc_norm",
        mode="max",
        save_top_k=1,
    )
    ckpt_last_callback = ModelCheckpoint(
        dirpath=f"experiments/{get_econfig_name(args)}",
        filename="last",
        every_n_train_steps=1000,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        logger=wandb_logger,
        default_root_dir=f"experiments/{get_econfig_name(args)}",
    )

    # Tune lr
    if not resume_ckpt:  # skip lr tuning if resuming from checkpoint
        tuner = Tuner(trainer)
        tuner.lr_find(model, dm)

    # Add evaluation callback after lr tuning
    trainer.callbacks.extend([eval_callback, ckpt_best_callback, ckpt_last_callback])
    trainer.fit(model, dm, ckpt_path=resume_ckpt)  # for auto resume, not for saving


if __name__ == "__main__":
    main()


# Epoch 0:   2%|██▉       | 5337/315209 [54:08<52:23:38,  1.64it/s, v_num=v874, train_loss_step=4.630]^C
# Epoch 0:  21%|████████████████▋           | 16418/78803 [2:55:56<11:08:31,  1.56it/s, v_num=m7ey, train_loss_step=3.710]slurmstepd: error: container_p_join: open failed for /var/opt/slurm/localstorage/7233753/.ns: No such file or directory
