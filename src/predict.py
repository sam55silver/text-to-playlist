from pandas import DataFrame
from src.model import PlaylistDescriptionRegressor, knn_audio_features
from transformers import DistilBertTokenizer


def get_recommendations(
    prompt: str,
    model: PlaylistDescriptionRegressor,
    nlp_model_name: str,
    audio_features: DataFrame,
    max_len: int,
    recommend_songs: int,
) -> list:
    # get tokenized prompt
    tokenizer = DistilBertTokenizer.from_pretrained(nlp_model_name)

    # Tokenize and encode the playlist description
    encoded_playlist_description = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = encoded_playlist_description["input_ids"]
    attention_mask = encoded_playlist_description["attention_mask"]

    # get the predicted audio features
    predicted_audio_features = model(input_ids, attention_mask)

    # Drop ids, names, artists, and uri from audio features and convert to numpy
    audio_features_no_id = audio_features.drop(
        columns=["id", "name", "artist", "uri"]
    ).to_numpy()

    # get the knn model
    knn = knn_audio_features(audio_features_no_id, recommend_songs)

    # get the neighbors
    _, pred_indices = knn.kneighbors(predicted_audio_features.detach().numpy())

    # get the ids of the recommend_songs neighbors
    neighbors = audio_features.iloc[pred_indices[0]][:recommend_songs]

    # get name, artist, and uri
    neighbors = neighbors[["name", "artist", "uri"]]
    # reset the index
    neighbors = neighbors.reset_index(drop=True)

    # loop through the neighbors and print the name artist and uri
    print("\nPrompt:", prompt)
    print("Playlist:")
    print(neighbors.to_markdown())
