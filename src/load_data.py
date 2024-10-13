import json
from pathlib import Path
from time import sleep

import numpy as np
import pandas as pd
import requests
from src.model import PlaylistDescriptionRegressor


def load_playlist_slice(slice: int) -> pd.DataFrame:
    first = slice * 1000
    last = first + 999

    # Load the slice as json
    data = json.load(
        open(
            Path(__file__).parents[1]
            / f"data/spotify_million_playlist_dataset/data/mpd.slice.{first}-{last}.json"
        )
    )

    # Convert the slice to a data frame
    return pd.DataFrame(data["playlists"])


def load_playlists(num_slices: int) -> tuple[pd.DataFrame, dict]:
    # Load the number of playlists specified by num_slices
    # Each slice is 1000 playlists
    all_tracks = {}
    slices = []
    for i in range(num_slices):
        # get slice dataframe
        df = load_playlist_slice(i)

        # Remove all columns except for the description and tracks
        df = df[["description", "tracks"]]

        # Get all playlists with descriptions
        df = df[df["description"].notna()]
        # Remove playlists with symbols in the description
        df = df[df["description"].str.contains("[^a-zA-Z0-9 ]", regex=True)]

        # Convert the tracks to a list of tracks
        tracks = df["tracks"].to_numpy()

        # Collect all track IDs and names
        tracks_ids = []
        for playlist in tracks:
            tracks = []
            for track in playlist:
                id = track["track_uri"].split(":")[2]
                name = track["track_name"]
                artist = track["artist_name"]

                tracks.append(id)

                all_tracks[id] = {"name": name, "artist": artist}

            tracks = np.array(tracks)
            tracks_ids.append(tracks)

        # Replace the tracks column with the list of tracks
        df["tracks"] = tracks_ids

        # Append the slice to the list of slices
        slices.append(df)

    # Concatenate the slices into a single data frame
    df = pd.concat(slices)

    # Reset the index
    df = df.reset_index(drop=True)

    return df, all_tracks


def generate_datasets(slices: int) -> None:
    # Load the playlists data in slices into a single data frame
    # num of playlists = 1000 * num_slices
    try:
        playlists, all_tracks = load_playlists(slices)
    except FileNotFoundError:
        print(
            "../data/spotify_million_playlist_dataset/data/mpd.slice.*.json not found. "
            + "Please download the million playlist dataset from "
            + "https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files"
            + " to generate new data."
        )
        exit()

    print("Finished loading playlist and track data!")

    # convert all_tracks to a list of track ids
    track_ids = list(all_tracks.keys())

    # Download the audio features for the tracks from the Spotify API
    audio_features = download_audio_features(track_ids, all_tracks)

    # Save to pkl
    playlists.to_pickle(
        Path(__file__).parents[1] / f"data/playlists_dataset_{slices}.pkl"
    )
    audio_features.to_pickle(
        Path(__file__).parents[1] / f"data/audio_features_{slices}.pkl"
    )


def download_audio_features(track_ids: list, all_tracks: dict) -> pd.DataFrame:
    # Calculate wait time
    num_tracks = len(all_tracks)
    num_requests = num_tracks // 100 + 1

    sleep_time = 1
    wait_time = sleep_time * num_requests
    print(
        f"Now downloading audio features for {num_tracks} tracks. This will take around {wait_time} seconds...\n"
    )

    # Get Client ID and Client Secret from JSON file **Not sharing this file**
    client = json.load(open(Path(__file__).parents[1] / "data/spotify_ids.json"))

    # Generate access token for Spotify API
    token = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client["id"],
            "client_secret": client["secret"],
        },
    ).json()["access_token"]

    # For every 100 tracks, make a request to the Spotify API to get the audio features for those tracks
    tracks_audio_feat = []
    for i in range(0, num_tracks, 100):
        # Get the tracks for the current request
        tracks_slice = track_ids[i : i + 100]
        print(f"Getting audio features for tracks {i} to {i + 99}...")

        try:
            # Get the audio features for the tracks
            req_audio_features = requests.get(
                "https://api.spotify.com/v1/audio-features",
                params={"ids": ",".join(tracks_slice)},
                headers={"Authorization": "Bearer {token}".format(token=token)},
            ).json()["audio_features"]

            print(f"Got audio features for tracks {i} to {i + 99}!")

            # Extract all important audio features into a dictionary
            audio_features = []
            for track in req_audio_features:
                # if track cannot be downloaded, skip it
                if track is None:
                    continue

                audio_features.append(
                    {
                        "id": track["id"],
                        "name": all_tracks[track["id"]]["name"],
                        "artist": all_tracks[track["id"]]["artist"],
                        "uri": track["uri"],
                        "danceability": track["danceability"],
                        "energy": track["energy"],
                        "loudness": track["loudness"],
                        "speechiness": track["speechiness"],
                        "acousticness": track["acousticness"],
                        "instrumentalness": track["instrumentalness"],
                        "liveness": track["liveness"],
                        "valence": track["valence"],
                        "tempo": track["tempo"],
                    }
                )

            # normalize loudness and tempo
            loudness = np.array([track["loudness"] for track in audio_features])
            loudness = (loudness - np.min(loudness)) / (
                np.max(loudness) - np.min(loudness)
            )
            tempo = np.array([track["tempo"] for track in audio_features])
            tempo = (tempo - np.min(tempo)) / (np.max(tempo) - np.min(tempo))

            # add normalized loudness and tempo to audio features
            for i, track in enumerate(audio_features):
                track["loudness"] = loudness[i]
                track["tempo"] = tempo[i]

            tracks_audio_feat += audio_features

        except Exception as e:
            print(f"Failed to get audio features for tracks {i} to {i + 99}!")
            print(f"Error: {e}")
            exit()

        # wait to avoid rate limiting
        print(f"Waiting for {sleep_time} seconds...")
        sleep(sleep_time)

    # Convert to df
    tracks = pd.DataFrame(tracks_audio_feat)

    return tracks


def load_playlists_dataset(slices: int) -> pd.DataFrame:
    # Load the playlists dataset
    return pd.read_pickle(
        Path(__file__).parents[1] / f"data/playlists_dataset_{slices}.pkl"
    )


def load_audio_features_dataset(slices: int) -> pd.DataFrame:
    # Load the audio features dataset
    return pd.read_pickle(
        Path(__file__).parents[1] / f"data/audio_features_{slices}.pkl"
    )


def load_model(
    slices: int,
    nlp_model_name: str,
    input_size: int,
    output_size: int,
    learn_rate: float,
    weight_decay: float,
) -> PlaylistDescriptionRegressor:
    # Load the model
    model = PlaylistDescriptionRegressor.load_from_checkpoint(
        Path(__file__).parents[1] / f"data/model_{slices}.ckpt",
        nlp_model_name=nlp_model_name,
        input_size=input_size,
        output_size=output_size,
        learn_rate=learn_rate,
        weight_decay=weight_decay,
    )

    model.eval()

    return model


def load_datasets(slices: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        playlists = load_playlists_dataset(slices)
        audio_features = load_audio_features_dataset(slices)
    except FileNotFoundError:
        print(
            f"../data/playlist_dataset_{slices}.pkl or ../data/audio_features_{slices}.pkl is not found."
        )
        print(
            "Data must not be generated. Please run the script with the --download flag to download the data. "
            + "Use --slices to choose the number of playlists to load."
        )
        print("Exiting...")
        exit()

    return playlists, audio_features
