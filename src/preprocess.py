import numpy as np
import pandas as pd
import torch


def get_avg_audio_features(
    playlists_tracks: np.ndarray, audio_features: pd.DataFrame
) -> list:
    # Set the index to the track id
    audio_features.set_index("id", inplace=True)

    playlist_avg_audio_features = []
    for playlist in playlists_tracks:
        # Dict to store the average audio features for the playlist
        avg_audio_features = {
            "danceability": 0,
            "energy": 0,
            "loudness": 0,
            "speechiness": 0,
            "acousticness": 0,
            "instrumentalness": 0,
            "liveness": 0,
            "valence": 0,
            "tempo": 0,
        }
        for track in playlist:
            # get the audio features for the track
            try:
                track_features = audio_features.loc[track]
            except KeyError:
                # skip the track if it doesn't have audio features
                continue

            # add the audio features to the average
            avg_audio_features["danceability"] += track_features["danceability"]
            avg_audio_features["energy"] += track_features["energy"]
            avg_audio_features["loudness"] += track_features["loudness"]
            avg_audio_features["speechiness"] += track_features["speechiness"]
            avg_audio_features["acousticness"] += track_features["acousticness"]
            avg_audio_features["instrumentalness"] += track_features["instrumentalness"]
            avg_audio_features["liveness"] += track_features["liveness"]
            avg_audio_features["valence"] += track_features["valence"]
            avg_audio_features["tempo"] += track_features["tempo"]

        # divide the sum by the number of tracks to get the average
        avg_audio_features["danceability"] /= len(playlist)
        avg_audio_features["energy"] /= len(playlist)
        avg_audio_features["loudness"] /= len(playlist)
        avg_audio_features["speechiness"] /= len(playlist)
        avg_audio_features["acousticness"] /= len(playlist)
        avg_audio_features["instrumentalness"] /= len(playlist)
        avg_audio_features["liveness"] /= len(playlist)
        avg_audio_features["valence"] /= len(playlist)
        avg_audio_features["tempo"] /= len(playlist)

        # Turn the dict into a numpy array
        avg_audio_features = [
            avg_audio_features["danceability"],
            avg_audio_features["energy"],
            avg_audio_features["loudness"],
            avg_audio_features["speechiness"],
            avg_audio_features["acousticness"],
            avg_audio_features["instrumentalness"],
            avg_audio_features["liveness"],
            avg_audio_features["valence"],
            avg_audio_features["tempo"],
        ]

        # turn list into a torch tensor
        avg_audio_features = torch.tensor(avg_audio_features, dtype=torch.float32)

        playlist_avg_audio_features.append(avg_audio_features)

    # turn index into a column called id
    audio_features.reset_index(inplace=True)

    return playlist_avg_audio_features
