import numpy as np
import pyloudnorm as pyln
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
import random


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
  torch.cuda.manual_seed(SEED)
  torch.cuda.manual_seed_all(SEED)

def init_model() -> nn.Module:
  device = "cuda" if torch.cuda.is_available() else "cpu"

  model = Autoencoder().to(device)
  # Input dimension: [B, 1, L]   (B = batch size, mono audio with variable length L)
  # Output dimension: [B, 1, L]  (denoised waveform, same length as input)

  return model


# Do not change function signature
def train_model(model: nn.Module, dev_dataset: Dataset) -> nn.Module:
  train_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True)

  trainer = pl.Trainer(
      max_epochs=10,
      log_every_n_steps=10,
      enable_checkpointing=False,
      logger=False
  )
  trainer.fit(model, train_loader)

  return model


def get_data():
  try:
    cmu_arctic = torchaudio.datasets.CMUARCTIC("data_scratch", download=False)
  except Exception:
    cmu_arctic = torchaudio.datasets.CMUARCTIC("data_scratch", download=True)

  train_size = int(len(cmu_arctic) * 0.8)
  test_size = len(cmu_arctic) - train_size
  train_arctic, test_arctic = torch.utils.data.random_split(cmu_arctic, [train_size, test_size])

  train_dataset = AudioDataset(train_arctic)
  test_dataset = AudioDataset(test_arctic)
  return train_dataset, test_dataset


def normalize_audio(waveform: torch.Tensor, sampling_rate: int = 16000) -> torch.Tensor:
  data = waveform.detach().cpu().numpy().transpose(1, 0)

  meter = pyln.Meter(sampling_rate)
  loudness = meter.integrated_loudness(data)

  if not np.isfinite(loudness):
    return waveform.float()

  normalized = pyln.normalize.loudness(data, loudness, -24.0)
  return torch.from_numpy(normalized).transpose(1, 0).float()


class AudioDataset(Dataset):
  def __init__(self, audio_files):
    self.audio_files = audio_files

  def __len__(self):
    return len(self.audio_files)

  def __getitem__(self, idx):
    waveform, sample_rate, *_ = self.audio_files[idx]

    if sample_rate != 16000:
      waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    target_length = 16000 * 2

    if waveform.size(1) < target_length:
      padding = target_length - waveform.size(1)
      waveform = torch.nn.functional.pad(waveform, (0, padding))

    if waveform.size(1) > target_length:
      start_idx = torch.randint(0, waveform.size(1) - target_length + 1, (1,)).item()
      waveform = waveform[:, start_idx:start_idx + target_length]

    noisy_waveform = waveform + torch.randn_like(waveform) * 0.1

    noisy_waveform = normalize_audio(noisy_waveform)
    waveform = normalize_audio(waveform)

    return noisy_waveform, waveform


class Snake(nn.Module):
  def __init__(self, alpha=1.0):
    super().__init__()
    self.alpha = nn.Parameter(torch.tensor(alpha))

  def forward(self, x):
    return x + (1.0 / self.alpha) * torch.pow(torch.sin(self.alpha * x), 2)


class Autoencoder(pl.LightningModule):
  def __init__(self):
    super().__init__()

    self.enc1 = nn.Sequential(
        nn.Conv1d(1, 64, kernel_size=7, stride=3, padding=1),
        Snake()
    )
    self.enc2 = nn.Sequential(
        nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU()
    )
    self.enc3 = nn.Sequential(
        nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU()
    )
    self.enc4 = nn.Sequential(
        nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=0),
        nn.LeakyReLU()
    )

    self.dec1 = nn.Sequential(
        nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=0),
        nn.LeakyReLU()
    )
    self.dec2 = nn.Sequential(
        nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU()
    )
    self.dec3 = nn.Sequential(
        nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU()
    )
    self.dec4 = nn.Sequential(
        nn.ConvTranspose1d(128, 1, kernel_size=7, stride=3, padding=1),
        nn.Tanh()
    )

  def forward(self, x):
    e1 = self.enc1(x)
    e2 = self.enc2(e1)
    e3 = self.enc3(e2)
    e4 = self.enc4(e3)

    d1 = self.dec1(e4)
    d1 = torch.cat([d1, e3], dim=1)

    d2 = self.dec2(d1)
    d2 = torch.cat([d2, e2], dim=1)

    d3 = self.dec3(d2)
    d3 = torch.cat([d3, e1], dim=1)

    x_hat = self.dec4(d3)
    return x_hat

  def training_step(self, batch, batch_idx):
    noisy_sample, ground_truth = batch

    x_hat = self(noisy_sample)

    if x_hat.size(-1) != ground_truth.size(-1):
      padding = ground_truth.size(-1) - x_hat.size(-1)
      x_hat = torch.nn.functional.pad(x_hat, (0, padding))

    recon_loss = nn.MSELoss()(ground_truth, x_hat)
    self.log("recon_loss", recon_loss, prog_bar=True)
    return recon_loss

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=0.001)


def test_model(model: nn.Module, test_dataset: Dataset) -> float:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
  loss_fn = nn.MSELoss()

  model = model.to(device)
  model.eval()

  total_loss = 0.0
  total_samples = 0

  with torch.no_grad():
    for noisy_sample, ground_truth in test_loader:
      noisy_sample = noisy_sample.to(device)
      ground_truth = ground_truth.to(device)

      x_hat = model(noisy_sample)

      if x_hat.size(-1) < ground_truth.size(-1):
        padding = ground_truth.size(-1) - x_hat.size(-1)
        x_hat = torch.nn.functional.pad(x_hat, (0, padding))
      elif x_hat.size(-1) > ground_truth.size(-1):
        x_hat = x_hat[..., :ground_truth.size(-1)]

      batch_size = noisy_sample.size(0)
      total_loss += loss_fn(x_hat, ground_truth).item() * batch_size
      total_samples += batch_size

  return total_loss / max(total_samples, 1)


def test() -> float:
  train_dataset, test_dataset = get_data()
  model = init_model()
  model = train_model(model, train_dataset)
  return test_model(model, test_dataset)


def run():
  train_dataset: Dataset
  test_dataset: Dataset
  train_dataset, test_dataset = get_data()

  model = init_model()
  model = train_model(model, train_dataset)

  model.eval()
  score = test_model(model, test_dataset)

  return score


if __name__ == "__main__":
  run()
