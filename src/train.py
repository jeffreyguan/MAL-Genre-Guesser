import sys
sys.path.append('src')
import torch
import kagglehub
import pandas as pd
import numpy as np
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torch import nn
from torchvision import datasets, transforms
import torchvision.models as models
from dataset import AnimeDataset
from model import AnimeGenreClassifier


# load data
path = kagglehub.dataset_download("dbdmobile/myanimelist-dataset")
df = pd.read_csv(f"{path}/anime-dataset-2023.csv")

# split by ','
all_genres = [genre.strip() for genres in df['Genres'] for genre in genres.split(',')]

# dont train on genres with under 200 shows or the 'UNKNOWN' genre
genre_counts = Counter(all_genres)
MIN_COUNT = 200
genres_to_keep = [genre for genre, count in genre_counts.most_common() 
    if count >= MIN_COUNT and genre not in ['UNKNOWN']]

# filter rows to only those with at least one valid genre
def has_valid_genre(genre_str):
    genres = [g.strip() for g in genre_str.split(',')]
    return any(g in genres_to_keep for g in genres)

df = df[df['Genres'].apply(has_valid_genre)]

# turn genres into a vector
def encode_genres(genre_str, genres_to_keep):
    genres = [g.strip() for g in genre_str.split(',')]
    vector = [1 if genre in genres else 0 for genre in genres_to_keep]
    return vector

df['genre_vector'] = df['Genres'].apply(lambda x: encode_genres(x, genres_to_keep))

# split into training data and validation/testing data
synopses = df['Synopsis'].tolist()
labels = df['genre_vector'].tolist()

train_synopses, test_synopses, train_labels, test_labels = train_test_split(
    synopses, labels, test_size=0.2, random_state=67
)

BATCH_SIZE = 32

training_data = AnimeDataset(train_synopses, train_labels)
test_data = AnimeDataset(test_synopses, test_labels)

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
model = AnimeGenreClassifier(num_genres=len(genres_to_keep)).to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, data in enumerate(dataloader):
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        # Compute prediction and loss
        pred = model(input_ids, attention_mask)
        loss = loss_fn(pred, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(input_ids)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    all_preds = []
    all_labels = []

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for data in dataloader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            pred = model(input_ids, attention_mask)
            test_loss += loss_fn(pred, labels).item()

            # apply sigmoid then threshold at 0.5
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            all_preds.extend(pred_binary.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss /= num_batches
    f1 = f1_score(np.array(all_labels), np.array(all_preds), average='macro')
    print(f"Test Error: \n Accuracy: {f1 * 100:.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 6
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")