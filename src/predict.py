import gradio as gr
import torch
from transformers import DistilBertTokenizer
from model import AnimeGenreClassifier

genres_to_keep = ['Comedy', 'Fantasy', 'Action', 'Adventure', 'Sci-Fi', 'Drama', 
                  'Romance', 'Slice of Life', 'Supernatural', 'Hentai', 'Mystery', 
                  'Avant Garde', 'Ecchi', 'Sports', 'Horror', 'Suspense', 'Award Winning']

model = AnimeGenreClassifier(num_genres=len(genres_to_keep))
model.load_state_dict(torch.load('./model.pth'))
model.eval()

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def predict(synopsis):
    encoding = tokenizer(synopsis, return_tensors='pt', truncation=True, max_length=256, padding='max_length')
    with torch.no_grad():
        output = model(encoding['input_ids'], encoding['attention_mask'])
    probs = torch.sigmoid(output).squeeze()
    predicted = [genres_to_keep[i] for i, p in enumerate(probs) if p > 0.5]
    return ', '.join(predicted)

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=10, placeholder="Paste anime synopsis here..."),
    outputs=gr.Textbox(label="Predicted Genres"),
    title="Anime Genre Classifier"
).launch()