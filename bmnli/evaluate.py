import torch
import numpy as np
from sklearn.metrics import f1_score


def predict(model, data_loader):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictions = []
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits = model(**batch).logits[:, 0]
        probs = torch.nn.Sigmoid()(logits).detach().cpu().numpy()
        predictions.append(probs)
    predictions = np.concatenate(predictions)
    return predictions


def evaluate(model, val_loader, labels):
    with torch.no_grad():
        pred = predict(model, val_loader)
        best_score = 0
        best_thresh = 0
        for t in np.linspace(pred.min(), pred.max(), num=200):
            pt = (pred < t).astype('float32')
            score = f1_score(labels, pt, average='macro')
            if score > best_score:
                best_score = score
                best_thresh = t
    return best_score, best_thresh
