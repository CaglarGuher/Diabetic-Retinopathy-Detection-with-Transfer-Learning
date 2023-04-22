import torch
from utils import get_predictions
import os
from utils import get_data,preprocess_image
import pandas as pd
from models import model_dict,select_model
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
dataset_total = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
validation_directiory = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"
train_test_image_directory = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

files_and_dirs = os.listdir("C:/Users/PC/Desktop/saved_models/") 
all_predictions = []
weights = []


for items in files_and_dirs:
    model_name, file_extension = os.path.splitext(items)

    _, _, valid_data = get_data(dataset_total, train_test_image_directory, validation_directiory,
                                train_test_sample_size=10, batch_size=16,
                                image_filter=preprocess_image, model=model_name, validation=True)

    state_dict = torch.load(f"C:/Users/PC/Desktop/saved_models/{items}")

    model_instance = select_model(model_name)

    model_instance = model_instance.to(device)

    model_instance.load_state_dict(state_dict)

    weight = model_dict[model_name]["weight"]
    weights.append(weight)

    predictions = get_predictions(model_instance, valid_data, device)
    all_predictions.append(predictions)


all_predictions = list(zip(*all_predictions))

ensemble_preds = []
for preds in all_predictions:
    weighted_votes = Counter()
    for pred, weight in zip(preds, weights):
        weighted_votes[pred] += weight
    ensemble_preds.append(weighted_votes.most_common(1)[0][0])

true_labels = []
for _, labels in valid_data:
    true_labels.extend(labels.numpy())


def evaluate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix
    }

print(ensemble_preds)