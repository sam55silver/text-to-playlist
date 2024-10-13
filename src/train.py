from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import KFold, train_test_split
from src.model import ModelLogger, PlaylistDataset, PlaylistDescriptionRegressor
from src.preprocess import get_avg_audio_features
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer


def train_model(
    playlists: pd.DataFrame,
    audio_features: pd.DataFrame,
    slices: int,
    epochs: int,
    k_folds: int,
    max_len: int,
    nlp_model_name: str,
    input_size: int,
    output_size: int,
    weight_decay: float,
    learn_rate: float,
) -> None:
    # Preprocess the data
    avg_audio_features = get_avg_audio_features(
        playlists["tracks"].to_numpy(), audio_features
    )

    # Create the data
    data = []
    for i, avg_tracks in enumerate(avg_audio_features):
        data.append((playlists["description"][i], avg_tracks))

    # Load the tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(nlp_model_name)

    # K-fold cross validation
    folds = []
    kf = KFold(n_splits=k_folds, shuffle=True)
    for train_index, test_index in kf.split(data):
        fold_num = len(folds) + 1

        # Split data into train and test sets
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        # Split train data into train and validation sets
        train_data, val_data = train_test_split(train_data, test_size=0.2)

        # Load the data into datasets
        train_dataloader = PlaylistDataset(train_data, tokenizer, max_len)
        val_dataloader = PlaylistDataset(val_data, tokenizer, max_len)
        test_dataloader = PlaylistDataset(test_data, tokenizer, max_len)

        # Create data loaders with datasets
        batch_size = 32
        num_workers = 4

        train_dataloader = DataLoader(
            train_dataloader,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        val_dataloader = DataLoader(
            val_dataloader, batch_size=batch_size, num_workers=num_workers
        )
        test_dataloader = DataLoader(
            test_dataloader, batch_size=batch_size, num_workers=num_workers
        )

        # Free up GPU memory
        torch.cuda.empty_cache()

        # Create the model
        model = PlaylistDescriptionRegressor(
            nlp_model_name, input_size, output_size, weight_decay, learn_rate
        )

        # Train the model
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            devices=1,
            logger=ModelLogger(learn_rate=learn_rate, weight_decay=weight_decay),
        )
        trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
        )

        # Test the model

        # Create lists to store all targets and predictions from batches
        all_targets = []
        all_predictions = []

        # Iterate over the test data
        with torch.no_grad():
            for batch in test_dataloader:
                # Extract variables from batch
                input_ids, attention_mask, avg_audio_features = batch

                # Perform prediction
                predicted_avg_audio_features = model(input_ids, attention_mask)

                # Add targets and predictions to lists
                all_targets.append(avg_audio_features.numpy())
                all_predictions.append(predicted_avg_audio_features.numpy())

        # Concatenate all targets and predictions
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)

        # mean squared error
        mse = np.mean((all_targets - all_predictions) ** 2)

        # Print the mean squared error
        print(f"Mean squared error: {mse}")

        # Save model and mse
        folds.append(
            {
                "fold": fold_num,
                "trainer": trainer,
                "mse": mse,
            }
        )

    # Create a df of the folds with index as the fold number
    folds = pd.DataFrame(folds)

    # Drop the trainer
    printable_folds = folds.drop(columns=["trainer"])

    # Print the folds
    print(printable_folds.to_markdown(index=False))

    # Sort the folds by mse
    folds = folds.sort_values(by="mse")

    # Get the best fold trailer
    best_fold = folds.iloc[0]
    trainer = best_fold["trainer"]

    # Save the model
    print("Saving most accurate model...")
    trainer.save_checkpoint(Path(__file__).parents[1] / f"data/model_{slices}.ckpt")
