import torch

from mtp.mheads._abc import (
    AbstractDisributionHead,
    AbstractDisributionHeadConfig,
    AbstractDisributionHeadOutput,
)


class STP(AbstractDisributionHead):
    def __init__(self, config: AbstractDisributionHeadConfig):
        super().__init__(config)
        self.head = torch.nn.Linear(config.d_model, config.d_output)

    def forward(self, x, y=None):
        logits = self.head(x)
        if y is not None:
            loss = torch.nn.functional.cross_entropy(logits, y)
            return AbstractDisributionHeadOutput(logits=logits, loss=loss)
        else:
            return AbstractDisributionHeadOutput(logits=logits)
