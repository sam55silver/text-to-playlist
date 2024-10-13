import numpy as np
import pandas as pd
from src.load_data import load_datasets, load_playlist_slice
from src.preprocess import get_avg_audio_features


def explore_data():
    # Get all 1000 slices of the dataset
    # Only retrieve the description and playlist names columns
    dfs = [load_playlist_slice(i)[["description", "name"]] for i in range(1000)]
    playlists = pd.concat(dfs)

    print("All Playlists: ", playlists.shape[0])

    # Get all playlists with no description
    playlists_names = playlists[playlists["description"].isna()]
    print("Playlists with only name: ", playlists_names.shape[0])

    # remove all playlists with symbols in the name
    playlists_names = playlists_names[
        playlists_names["name"].str.contains("[^a-zA-Z0-9 ]", regex=True)
    ]
    print("Playlists with only letters in name: ", playlists_names.shape[0])

    # Remove playlists with no description
    playlists_desc = playlists[playlists["description"].notna()]
    print("Playlists with description: ", playlists_desc.shape[0])

    # Remove playlists with symbols in the description
    playlists_desc = playlists_desc[
        playlists_desc["description"].str.contains("[^a-zA-Z0-9 ]", regex=True)
    ]
    print("Playlists with only letters in description: ", playlists_desc.shape[0])

    # Get average word length of playlist names
    playlists["name_len"] = playlists["name"].apply(lambda x: len(x.split()))
    print("Average playlist name length: ", playlists["name_len"].mean())

    # Get average word length of playlist descriptions
    playlists_desc["desc_len"] = playlists_desc["description"].apply(
        lambda x: len(x.split())
    )
    print("Average playlist description length: ", playlists_desc["desc_len"].mean())

    # Playlists with only emoji in name
    playlists_emoji = playlists[playlists["name"].apply(lambda x: x.isascii())]
    print("Playlists with only emoji in name: ", playlists_emoji.shape[0])

    # Get all features of the playlist dataset
    single_playlist = load_playlist_slice(0)
    print("\nPlaylist Columns: ")
    for col in single_playlist.columns:
        print(col)

    # Get all features of the track dataset
    single_track = single_playlist["tracks"][0][0]
    print("\nTrack Columns: ")
    for col in single_track.keys():
        print(col)

    # Load playlist and audio features
    full_playlist, audio_features = load_datasets(1000)

    # print unique songs shape
    print("Unique Songs:", audio_features.shape[0])

    # Get the describe of the full data frame

    # get averages
    avg_audio_features = get_avg_audio_features(
        full_playlist["tracks"].to_numpy(), audio_features
    )

    # Convert tensors to numpy
    data = []
    for playlist in avg_audio_features:
        non_tensor = np.array(playlist)
        data.append(non_tensor)

    # convert to data frame
    data = pd.DataFrame(data)

    # Run describe and rename columns
    data_describe = data.iloc[:, 0:].astype(float).describe()
    data_describe = data_describe.rename(
        columns={
            0: "danceability",
            1: "energy",
            2: "loudness",
            3: "speechiness",
            4: "acousticness",
            5: "instrumentalness",
            6: "liveness",
            7: "valence",
            8: "tempo",
        }
    )

    # Display
    print("\nFull Dataset Described:")
    print(data_describe.round(3).to_markdown())
