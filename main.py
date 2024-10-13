import argparse

import src.load_data as data
from src.explore_data import explore_data
from src.predict import get_recommendations
from src.train import train_model

EPOCHS = 10
K_FOLDS = 5

MAX_LEN = 128
NLP_MODEL_NAME = "distilbert-base-uncased"
INPUT_SIZE = 768
OUTPUT_SIZE = 9
RECOMMEND_SONGS = 5
WEIGHT_DECAY = 1e-3
LEARN_RATE = 1e-5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument(
        "--download",
        help="Download data from spotify. "
        + "This will require a spotify api access id and secret and will take some time.",
        action="store_true",
    )
    parser.add_argument(
        "--train",
        help="Train the model using data with specified number of slices.",
        action="store_true",
    )
    parser.add_argument(
        "--explore",
        help="Run functions to explore the data.",
        action="store_true",
    )
    parser.add_argument(
        "--prompt",
        help="Use predict mode to predict the prompt inputted.",
        type=str,
    )
    parser.add_argument(
        "--slices",
        help="Number of playlists to use when generating data. 1000 playlists per slice.",
        type=int,
        default=50,
    )

    # Parse the arguments
    args = parser.parse_args()

    # Check if slices is a valid number
    if args.slices <= 0:
        print("Slices must be a positive integer. Exiting...")
        exit()
    elif args.slices > 1000:
        print("Slices must be less than or equal to 1000. Exiting...")
        exit()

    # Explore the data
    if args.explore:
        explore_data()

    # Download the data if the user wants to
    if args.download:
        data.generate_datasets(args.slices)

    if args.train:
        # Load the data
        playlists, audio_features = data.load_datasets(args.slices)

        train_model(
            playlists,
            audio_features,
            args.slices,
            epochs=EPOCHS,
            k_folds=K_FOLDS,
            max_len=MAX_LEN,
            nlp_model_name=NLP_MODEL_NAME,
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            learn_rate=LEARN_RATE,
            weight_decay=WEIGHT_DECAY,
        )

    # Predict the input if the user wants predict mode
    if args.prompt:
        # Load the data
        audio_features = data.load_audio_features_dataset(args.slices)

        # Check if input is a empty string
        if args.prompt == "":
            print("Prompt is empty. Please enter a string prompt as input. Exiting...")
            exit()
        else:
            # Check if a model is found
            try:
                model = data.load_model(
                    args.slices,
                    NLP_MODEL_NAME,
                    INPUT_SIZE,
                    OUTPUT_SIZE,
                    LEARN_RATE,
                    WEIGHT_DECAY,
                )
            except FileNotFoundError:
                print(
                    f"No model found for {args.slices} slices. Please run script with "
                    + "--train flag to train a model."
                )

            # Predict the input
            get_recommendations(
                args.prompt,
                model,
                NLP_MODEL_NAME,
                audio_features,
                MAX_LEN,
                RECOMMEND_SONGS,
            )
