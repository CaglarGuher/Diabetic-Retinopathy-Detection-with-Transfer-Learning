
'''
all_predictions = []
weights = []

# Get predictions from each model and store their weights
for model_dict in best_results:
    model = model_dict["model"]
    weight = model_dict["weight"]
    weights.append(weight)
    predictions = evaluate_model(model, dataloader, device)  # Assuming this function returns a list of class predictions
    all_predictions.append(predictions)

# Transpose the list of predictions so that each inner list contains predictions for the same sample from all models
all_predictions = list(zip(*all_predictions))

# Perform weighted voting
ensemble_preds = []
for preds in all_predictions:
    weighted_votes = Counter()
    for pred, weight in zip(preds, weights):
        weighted_votes[pred] += weight
    ensemble_preds.append(weighted_votes.most_common(1)[0][0])


evaluate_classification_metrics()
'''