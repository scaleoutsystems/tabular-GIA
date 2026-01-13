import torch
from torch import nn


class TabularMLP2(nn.Module):
	"""Simple two-layer MLP."""

	def __init__(self, d_in: int, d_hidden: int = 64, num_classes: int = 2) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(d_in, d_hidden),
			nn.ReLU(),
			nn.Linear(d_hidden, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)

class TabularMLP(nn.Module):
	"""Deeper MLP with normalization and dropout."""

	def __init__(self, d_in: int, d_hidden: int = 256, num_classes: int = 2, dropout: float = 0.1) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(d_in, d_hidden),
			nn.LayerNorm(d_hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_hidden, d_hidden),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(d_hidden, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)