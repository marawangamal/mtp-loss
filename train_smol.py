"""Minimal training script for SmolLM.

Usage:
    python train_smol.py --model_name distilbert/distilgpt2 --dataset_name wikitext --max_length 32 --epochs 1 --batch_size 1
"""

import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.data.data_collator import DataCollatorForLanguageModeling
import datasets
import lm_eval
from lm_eval import simple_evaluate
import wandb
from pytorch_lightning.loggers import WandbLogger


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
        # "streaming": True,
    },
}


class LMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        dataset_name,
        batch_size,
        max_length,
    ):
        super().__init__()
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
        self.dataset = datasets.load_dataset(**PRETRAINING_DS_CONFIG[self.dataset_name])
        self.dataset = self.dataset.filter(
            lambda x: x["text"] and x["text"].strip() != ""
        )
        self.dataset = self.dataset.shuffle(seed=42)
        # self.tokenized = self.dataset.map(
        #     lambda x: self.tokenizer(
        #         x["text"],
        #         truncation=True,
        #         max_length=self.max_length,
        #         return_tensors=None,
        #     ),
        #     batched=True,
        #     remove_columns=self.dataset.features.keys(),
        # )

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


class LitLM(pl.LightningModule):
    def __init__(self, model_name):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_config(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=5e-5)


class HellaSwagEvalCallback(pl.Callback):
    def __init__(self, model_name, eval_every_n_epochs=1, device=None):
        super().__init__()
        self.model_name = model_name
        self.eval_every_n_epochs = eval_every_n_epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if (epoch + 1) % self.eval_every_n_epochs == 0:
            print(
                f"\n[HellaSwagEvalCallback] Evaluating on HellaSwag at epoch {epoch+1}..."
            )
            results = simple_evaluate(
                model=lm_eval.models.huggingface.HFLM(pretrained=pl_module.model),
                tasks=["hellaswag"],
                num_fewshot=0,
                batch_size=2,
                gen_kwargs={"max_new_tokens": 40},
                # limit=100,
            )
            print(
                f"[HellaSwagEvalCallback] HellaSwag results: {results['results']['hellaswag']}"
            )
            # Log HellaSwag accuracy to wandb
            acc = acc_norm = results["results"]["hellaswag"].get("acc,none")
            acc_norm = results["results"]["hellaswag"].get("acc_norm,none")
            if acc_norm is not None:
                log_dict = {
                    "hellaswag/acc": acc,
                    "hellaswag/acc_norm": acc_norm,
                    "epoch": epoch + 1,
                }
                if (
                    hasattr(trainer.logger, "experiment")
                    and trainer.logger.experiment is not None
                ):
                    trainer.logger.experiment.log(log_dict)


def get_econfig_name(args: argparse.Namespace):
    name = ""
    for k, v in args.__dict__.items():
        name += f"-{k[:1]}{v}"
    return name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    p.add_argument("--dataset_name", type=str, default="fineweb")
    p.add_argument("--dataset_config", type=str, default="edu")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    wandb_logger = WandbLogger(project="mtl", name=get_econfig_name(args))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    dm = LMDataModule(tokenizer, args.dataset_name, args.batch_size, args.max_length)
    model = LitLM(args.model_name)
    eval_callback = HellaSwagEvalCallback(args.model_name, eval_every_n_epochs=1)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[eval_callback],
        logger=wandb_logger,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
