"""
data_loader.py: Data Preprocessing, PyTorch Dataset, and Log Corruption Pipeline
Author: A. Musab Ari
Project: YZM5110 Research Project - Multimodal Early Warning and Anomaly Detection in Uptime Applications

Handles:
- Server Machine Dataset (SMD) telemetry reading & normalization
- Log event tokenization & vocabulary indexing
- Strict chronological 70% / 10% / 20% data splitting (preventing temporal lookahead bias)
- Simulating syslog daemon crashes via controlled log corruption / dropout (e.g. 50% dropped logs)
"""

import os
import csv
import random
from typing import Tuple, List, Dict, Any

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SimpleLogTokenizer:
    """Lightweight rule-based tokenizer for unstructured system log messages."""
    def __init__(self, max_vocab_size: int = 5000):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<MASK>": 2}
        self.inv_vocab = {0: "<PAD>", 1: "<UNK>", 2: "<MASK>"}
        self.max_vocab_size = max_vocab_size

    def build_vocab(self, texts: List[str]):
        for text in texts:
            tokens = text.lower().replace(":", " ").replace("=", " ").replace(",", " ").split()
            for token in tokens:
                if token not in self.vocab and len(self.vocab) < self.max_vocab_size:
                    idx = len(self.vocab)
                    self.vocab[token] = idx
                    self.inv_vocab[idx] = token

    def encode(self, text: str, max_len: int = 24) -> Tuple[List[int], List[int]]:
        if not text or text.strip() == "":
            # Return padding if empty
            return [0] * max_len, [0] * max_len
        tokens = text.lower().replace(":", " ").replace("=", " ").replace(",", " ").split()
        ids = [self.vocab.get(t, 1) for t in tokens[:max_len]]
        mask = [1] * len(ids)
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids += [0] * pad_len
            mask += [0] * pad_len
        return ids, mask


if TORCH_AVAILABLE:
    class SMDMultimodalDataset(Dataset):
        """
        PyTorch Dataset for Server Machine Dataset (SMD) paired telemetry metrics and logs.
        """
        def __init__(
            self,
            metrics_data: List[List[float]],
            logs_data: List[str],
            labels: List[int],
            tokenizer: SimpleLogTokenizer,
            seq_len: int = 15,
            corrupt_logs: bool = False,
            corruption_rate: float = 0.5,
            seed: int = 42
        ):
            self.metrics_data = metrics_data
            self.logs_data = logs_data
            self.labels = labels
            self.tokenizer = tokenizer
            self.seq_len = seq_len
            self.corrupt_logs = corrupt_logs
            self.corruption_rate = corruption_rate
            self.rng = random.Random(seed)

            # Pre-compute sliding windows
            self.samples = []
            for i in range(len(metrics_data) - seq_len + 1):
                window_metrics = metrics_data[i:i + seq_len]  # (seq_len, num_features)
                current_log = logs_data[i + seq_len - 1]
                label = labels[i + seq_len - 1]
                self.samples.append((window_metrics, current_log, label))

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            window_metrics, current_log, label = self.samples[idx]

            # Simulate syslog crash / corrupted condition
            if self.corrupt_logs and (self.rng.random() < self.corruption_rate):
                # Dropped log message (empty/padding)
                log_text = ""
            else:
                log_text = current_log

            token_ids, mask = self.tokenizer.encode(log_text, max_len=24)

            # Transpose metrics to shape (num_features, seq_len) for 1D-CNN
            # window_metrics is (seq_len, num_features)
            num_features = len(window_metrics[0])
            metrics_transposed = [
                [window_metrics[t][f] for t in range(self.seq_len)]
                for f in range(num_features)
            ]

            return (
                torch.tensor(metrics_transposed, dtype=torch.float32),
                torch.tensor(token_ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long)
            )


def load_smd_csv(csv_path: str):
    """Reads the CSV file and returns parsed arrays."""
    timestamps = []
    metrics = []
    logs = []
    labels = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(int(row.get('timestamp', 0)))
            feat = [
                float(row.get('cpu_util', 0.0)),
                float(row.get('mem_util', 0.0)),
                float(row.get('disk_io_read', 0.0)),
                float(row.get('disk_io_write', 0.0)),
                float(row.get('net_in', 0.0)),
                float(row.get('net_out', 0.0)),
                float(row.get('load_1m', 0.0)),
                float(row.get('temp_c', 0.0))
            ]
            metrics.append(feat)
            logs.append(row.get('log_text', ''))
            labels.append(int(row.get('label', 0)))

    # Min-Max Normalization of metrics
    if metrics:
        num_features = len(metrics[0])
        for f in range(num_features):
            col_vals = [row[f] for row in metrics]
            c_min, c_max = min(col_vals), max(col_vals)
            diff = c_max - c_min if (c_max - c_min) > 1e-6 else 1.0
            for row in metrics:
                row[f] = (row[f] - c_min) / diff

    return timestamps, metrics, logs, labels


def get_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    seq_len: int = 15,
    corrupt_logs: bool = False,
    corruption_rate: float = 0.5,
    seed: int = 42
):
    """
    Creates chronological train (70%), validation (10%), and test (20%) DataLoaders.
    Strict chronological order prevents temporal lookahead bias in time-series anomaly detection.
    """
    timestamps, metrics, logs, labels = load_smd_csv(csv_path)
    n = len(metrics)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)

    # Train vocabulary only on training set to avoid data leakage
    tokenizer = SimpleLogTokenizer()
    tokenizer.build_vocab(logs[:train_end])

    if not TORCH_AVAILABLE:
        print("[data_loader.py] PyTorch not installed. Returning raw split dict.")
        return {
            "total_rows": n,
            "train_rows": train_end,
            "val_rows": val_end - train_end,
            "test_rows": n - val_end,
            "tokenizer_vocab_size": len(tokenizer.vocab),
            "corrupt_logs": corrupt_logs,
            "corruption_rate": corruption_rate
        }

    train_ds = SMDMultimodalDataset(
        metrics[:train_end], logs[:train_end], labels[:train_end],
        tokenizer, seq_len=seq_len, corrupt_logs=False, seed=seed
    )
    val_ds = SMDMultimodalDataset(
        metrics[train_end:val_end], logs[train_end:val_end], labels[train_end:val_end],
        tokenizer, seq_len=seq_len, corrupt_logs=False, seed=seed
    )
    test_ds = SMDMultimodalDataset(
        metrics[val_end:], logs[val_end:], labels[val_end:],
        tokenizer, seq_len=seq_len, corrupt_logs=corrupt_logs, corruption_rate=corruption_rate, seed=seed
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer


if __name__ == "__main__":
    test_path = os.path.join(os.path.dirname(__file__), "..", "sample_data", "smd_mini_sample.csv")
    print(f"[data_loader.py] Testing dataset loader on: {test_path}")
    if os.path.exists(test_path):
        res = get_dataloaders(test_path, batch_size=16)
        if TORCH_AVAILABLE:
            train_loader, val_loader, test_loader, tok = res
            print(f"DataLoaders initialized: Train batches={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
            batch = next(iter(train_loader))
            m, ids, mask, y = batch
            print(f"Batch shapes: metrics={m.shape}, ids={ids.shape}, mask={mask.shape}, y={y.shape}")
        else:
            print(f"Summary info: {res}")
        print("[data_loader.py] Dataset pipeline validated successfully.")
    else:
        print("[data_loader.py] Sample CSV not found at test path.")
