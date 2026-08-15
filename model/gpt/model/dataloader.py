import sys
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.utils.data

# Project root on sys.path
# Makes `utils` importable regardless of how / from where the script is run.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.file_utils import load_config  # noqa: E402  (import after path fix)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Data loading
def _load_tensors(cfg: dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load the pretraining ``X`` (input) and ``y`` (target) tensors from disk.

    Paths are constructed from the ``pretraining_data`` section of the config:

    * ``pt_data_dir`` – directory that holds the ``.pt`` files
    * ``x_file``      – filename for the input tensor (e.g. ``"X.pt"``)
    * ``y_file``      – filename for the target tensor (e.g. ``"y.pt"``)

    Args:
        cfg (dict): Parsed project configuration dictionary.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(X, y)`` tensors loaded with
        ``weights_only=True`` for security.

    Raises:
        FileNotFoundError: If either tensor file is missing.
    """
    pt_cfg = cfg["pretraining_data"]

    # Resolve relative paths against the project root so the script works
    # regardless of the current working directory.
    data_dir = PROJECT_ROOT / pt_cfg["pt_data_dir"]
    x_path = data_dir / pt_cfg["x_file"]
    y_path = data_dir / pt_cfg["y_file"]

    for path in (x_path, y_path):
        if not path.exists():
            raise FileNotFoundError(f"Tensor file not found: {path}")

    logger.info("Loading X  from: %s", x_path)
    X = torch.load(str(x_path), weights_only=True)

    logger.info("Loading y  from: %s", y_path)
    y = torch.load(str(y_path), weights_only=True)

    logger.info("X shape: %s  dtype: %s", X.shape, X.dtype)
    logger.info("y shape: %s  dtype: %s", y.shape, y.dtype)

    return X, y


def _split_tensors(
    X: torch.Tensor,
    y: torch.Tensor,
    cfg: dict,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor],
]:
    """
    Deterministically split ``(X, y)`` into train / validation / test subsets.

    Split fractions come from the ``dataloader`` section of the config:

    * ``train_split``      – fraction of data used for training (e.g. 0.8)
    * ``validation_split`` – fraction used for validation      (e.g. 0.1)
    * ``test_split``       – remainder goes to the test set

    The split is done on contiguous index ranges (no random sampling) so it
    is fully deterministic and reproducible without setting a random seed.

    Args:
        X (torch.Tensor): Input token-id tensor of shape ``(N, block_size)``.
        y (torch.Tensor): Target token-id tensor of shape ``(N, block_size)``.
        cfg (dict): Parsed project configuration dictionary.

    Returns:
        Three ``(X_split, y_split)`` tuples for train, validation, and test.

    Raises:
        ValueError: If any split ends up empty.
    """
    dl_cfg = cfg["dataloader"]
    n = len(X)

    # Cumulative index boundaries — NOT independent per-split sizes
    train_end = int(dl_cfg["train_split"] * n)
    val_end = train_end + int(dl_cfg["validation_split"] * n)
    # test slice: [val_end : n] (absorbs any rounding remainder)

    splits = {
        "train":      (X[:train_end],    y[:train_end]),
        "validation": (X[train_end:val_end], y[train_end:val_end]),
        "test":       (X[val_end:],      y[val_end:]),
    }

    for name, (Xs, ys) in splits.items():
        if len(Xs) == 0:
            raise ValueError(
                f"The '{name}' split is empty (n={n}, train_end={train_end}, "
                f"val_end={val_end}). Adjust split fractions in config.yml."
            )
        logger.info("%-12s  X: %s  y: %s", name, Xs.shape, ys.shape)

    return splits["train"], splits["validation"], splits["test"]


# DataLoader factories  
def _make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> torch.utils.data.DataLoader:
    """
    Wrap a ``(X, y)`` pair in a :class:`~torch.utils.data.TensorDataset` and
    return a configured :class:`~torch.utils.data.DataLoader`.

    Args:
        X (torch.Tensor): Input tensor for this split.
        y (torch.Tensor): Target tensor for this split.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle samples each epoch.
        num_workers (int): Subprocess workers for parallel data loading.
        pin_memory (bool): If ``True``, tensors are copied to pinned memory
            for faster CPU→GPU transfer.

    Returns:
        torch.utils.data.DataLoader: Ready-to-iterate DataLoader.
    """
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_dataloaders(
    cfg: dict | None = None,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """
    Build and return the train, validation, and test DataLoaders.

    This is the **primary public API** of this module.  All configuration is
    read from ``config/config.yml`` unless an already-parsed ``cfg`` dict is
    passed in (useful for testing or overrides).

    Relevant config keys (under the ``dataloader`` section):

    * ``batch_size``         – samples per mini-batch
    * ``shuffle_train``      – shuffle training set each epoch
    * ``shuffle_validation`` – shuffle validation set (usually ``false``)
    * ``shuffle_test``       – shuffle test set (usually ``false``)
    * ``num_workers``        – parallel loader sub-processes
    * ``pin_memory``         – pin tensors for faster GPU transfer

    Args:
        cfg (dict | None): Optional pre-loaded config dict.  Loads from disk
            when ``None``.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]:
            ``(train_loader, val_loader, test_loader)``
    """
    if cfg is None:
        cfg = load_config()

    dl_cfg = cfg["dataloader"]

    X, y = _load_tensors(cfg)
    (train_X, train_y), (val_X, val_y), (test_X, test_y) = _split_tensors(X, y, cfg)

    train_loader = _make_loader(
        train_X, train_y,
        batch_size=dl_cfg["batch_size"],
        shuffle=dl_cfg["shuffle_train"],
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
    )

    val_loader = _make_loader(
        val_X, val_y,
        batch_size=dl_cfg["batch_size"],
        shuffle=dl_cfg["shuffle_validation"],
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
    )

    test_loader = _make_loader(
        test_X, test_y,
        batch_size=dl_cfg["batch_size"],
        shuffle=dl_cfg["shuffle_test"],
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
    )

    logger.info(
        "DataLoaders ready — train: %d batches | val: %d batches | test: %d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader


# Smoke test
if __name__ == "__main__":
    logger.info("Running dataloader smoke test …")

    train_loader, val_loader, test_loader = get_dataloaders()

    # Peek at one batch from each split to confirm shapes are sensible
    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        x_batch, y_batch = next(iter(loader))
        logger.info(
            "[%s] batch x: %s  y: %s  x[0][:8]: %s",
            name, x_batch.shape, y_batch.shape, x_batch[0][:8].tolist(),
        )

    logger.info("Smoke test passed ✓")
