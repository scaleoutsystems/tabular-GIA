import torch
from torch import nn


class TabularMLP1(nn.Module):
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

class TabularMLP2(nn.Module):
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


class ResidualMLP(nn.Module):
	"""Residual block for tabular MLP."""

	def __init__(self, dim: int, dropout: float) -> None:
		super().__init__()
		self.net = nn.Sequential(
			nn.LayerNorm(dim),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(dim, dim),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return x + self.net(x)


class TabularMLP(nn.Module):
	"""Stronger MLP with residual blocks."""

	def __init__(self, d_in: int, d_hidden: int = 512, num_classes: int = 2, dropout: float = 0.3, n_blocks: int = 3) -> None:
		super().__init__()
		self.in_proj = nn.Linear(d_in, d_hidden)
		self.blocks = nn.Sequential(*[ResidualMLP(d_hidden, dropout) for _ in range(n_blocks)])
		self.head = nn.Sequential(
			nn.LayerNorm(d_hidden),
			nn.GELU(),
			nn.Dropout(dropout),
			nn.Linear(d_hidden, num_classes),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.in_proj(x)
		x = self.blocks(x)
		return self.head(x)
