import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from transformers import AutoModel


class PlaylistDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        playlist_description, avg_audio_features = self.data[idx]

        # Tokenize and encode the playlist description
        encoded_playlist_description = self.tokenizer(
            playlist_description,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encoded_playlist_description["input_ids"].squeeze(0)
        attention_mask = encoded_playlist_description["attention_mask"].squeeze(0)

        return input_ids, attention_mask, avg_audio_features


class PlaylistDescriptionRegressor(pl.LightningModule):
    def __init__(
        self, nlp_model_name, input_size, output_size, weight_decay, learn_rate
    ):
        super(PlaylistDescriptionRegressor, self).__init__()
        self.nlp_model = AutoModel.from_pretrained(nlp_model_name)
        self.regressor = nn.Sequential(
            nn.Linear(input_size, output_size, bias=True),
            nn.Sigmoid(),
        )
        self.learn_rate = learn_rate
        self.alpha = weight_decay

    def forward(self, input_ids, attention_mask):
        description_embedding = self.nlp_model(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state.mean(dim=1)
        return self.regressor(description_embedding)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), weight_decay=self.alpha, lr=self.learn_rate
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, avg_audio_features = batch
        predicted_avg_audio_features = self(input_ids, attention_mask)
        loss = nn.MSELoss()(predicted_avg_audio_features, avg_audio_features)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, avg_audio_features = batch
        predicted_avg_audio_features = self(input_ids, attention_mask)
        loss = nn.MSELoss()(predicted_avg_audio_features, avg_audio_features)
        self.log("val_loss", loss)


class ModelLogger(TensorBoardLogger):
    def __init__(self, learn_rate, weight_decay):
        super().__init__(
            "lightning_logs", name="lr_{}_wd_{}".format(learn_rate, weight_decay)
        )
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay


# Knn model
def knn_audio_features(targets: np.ndarray, k: int) -> NearestNeighbors:
    # Create knn model to find the closest audio features
    knn = NearestNeighbors(
        n_neighbors=k,
        algorithm="ball_tree",
        metric="mahalanobis",
        metric_params={"V": np.cov(targets.T)},
    )
    knn.fit(targets)

    return knn
