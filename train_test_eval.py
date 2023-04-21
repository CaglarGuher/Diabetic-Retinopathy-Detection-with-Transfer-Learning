import torch
from sklearn.metrics import confusion_matrix
import numpy as np



def train(dataloader,model, loss_fn, optimizer, device):
        
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss   = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total

        print('Train Loss: {:.4f} | Train Accuracy: {:.2f}%'.format(avg_loss, accuracy))

        return avg_loss, accuracy


def test(dataloader,model,loss_fn, device):

    model.eval() 
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): 
        
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)  # Move data to GPU
            
            output        = model(x)
            loss          = loss_fn(output, y).item()
            running_loss += loss
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    print('Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'.format(avg_loss, accuracy))

    return avg_loss, accuracy



def eval(model,device,dataloader):

    model.eval()

    # Create empty lists to store predictions and labels
    all_preds = []
    all_labels = []

    # Iterate over the validation set and make predictions

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print the confusion matrix

    print(cm)

    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    print(f"accuracy of the model is  = {accuracy}")
