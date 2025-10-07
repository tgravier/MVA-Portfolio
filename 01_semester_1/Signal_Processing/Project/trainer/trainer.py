import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from trainer.base_trainer import BaseTrainer

from util.utils import compute_STOI, compute_PESQ

class Trainer(BaseTrainer):
    def __init__(
            self,
            config,
            resume: bool,
            model,
            loss_function,
            optimizer,
            train_dataloader,
            validation_dataloader
    ):
        super(Trainer, self).__init__(config, resume, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0
        for i, (mixture, clean,name) in enumerate(self.train_dataloader):
            mixture = mixture.to(self.device)
            clean = clean.to(self.device)

            self.optimizer.zero_grad()
            enhanced = self.model(mixture)
            loss = self.loss(enhanced, clean)
            loss.backward()
            self.optimizer.step()

            loss_total += loss.item()

    @torch.no_grad()
    def _validation_epoch(self, epoch):


        sample_length = self.validation_custom_config["sample_length"]


        stoi_c_n = []  # clean and noisy
        stoi_c_e = []  # clean and enhanced
        pesq_c_n = []
        pesq_c_e = []

        for i, (mixture, clean, name) in enumerate(self.validation_dataloader):
            assert len(name) == 1, "Only support batch size is 1 in enhancement stage."
            name = name[0]
            padded_length = 0

            mixture = mixture.to(self.device)  # [1, 1, T]

            # The input of the model should be fixed length.
            if mixture.size(-1) % sample_length != 0:
                padded_length = sample_length - (mixture.size(-1) % sample_length)
                mixture = torch.cat([mixture, torch.zeros(1, 1, padded_length, device=self.device)], dim=-1)

            assert mixture.size(-1) % sample_length == 0 and mixture.dim() == 3
            mixture_chunks = list(torch.split(mixture, sample_length, dim=-1))

            enhanced_chunks = []
            for chunk in mixture_chunks:
                enhanced_chunks.append(self.model(chunk).detach().cpu())

            enhanced = torch.cat(enhanced_chunks, dim=-1)  # [1, 1, T]
            enhanced = enhanced if padded_length == 0 else enhanced[:, :, :-padded_length]
            mixture = mixture if padded_length == 0 else mixture[:, :, :-padded_length]

            enhanced = enhanced.reshape(-1).numpy()
            clean = clean.numpy().reshape(-1)
            mixture = mixture.cpu().numpy().reshape(-1)

            assert len(mixture) == len(enhanced) == len(clean)



            # Metric
            try:
                stoi_c_n.append(compute_STOI(clean, mixture, sr=16000))
                stoi_c_e.append(compute_STOI(clean, enhanced, sr=16000))
                pesq_c_n.append(compute_PESQ(clean, mixture, sr=16000))
                pesq_c_e.append(compute_PESQ(clean, enhanced, sr=16000))
            except Exception as e:
                print(f"Error calculating metrics for {name}: {e}")
                continue

        get_metrics_ave = lambda metrics: np.sum(metrics) / len(metrics)


        score = (get_metrics_ave(stoi_c_e) + self._transform_pesq_range(get_metrics_ave(pesq_c_e))) / 2
        return score