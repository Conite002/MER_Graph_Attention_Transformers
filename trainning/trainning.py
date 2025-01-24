import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from utils.debug import debug_message

def train_model(model, train_loader, val_loader, optimizer, criterion, args, epochs=10):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1} Training") as t:
            for features, labels in train_loader:
                features = features.to(args['device'])
                labels = labels.to(args['device'])

                optimizer.zero_grad()
                out = model(features, lengths=torch.tensor([len(features)]).to(args['device']))
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(args['device'])
                labels = labels.to(args['device'])

                out = model(features, lengths=torch.tensor([len(features)]).to(args['device']))
                val_loss += criterion(out, labels).item()

        val_loss /= len(val_loader)
        debug_message(f"Epoch {epoch + 1} train loss {train_loss / len(train_loader)}, val loss {val_loss}")
        # print accuracy and f1 score
        evaluate_model(model, val_loader, args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            # debug_message(f"Epoch {epoch + 1}: Best model saved")
        

def evaluate_model(model, test_loader, args):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        with tqdm(test_loader, desc="Testing") as t:
            for features, labels in t:
                features = features.to(args['device'])
                labels = labels.to(args['device'])

                out = model(features, lengths=torch.tensor([len(features)]).to(args['device']))
                preds = out.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    test_acc = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='macro')

    debug_message(f"Test Accuracy: {test_acc * 100:.2f}%, Test F1 Score: {test_f1 * 100:.2f}%")

    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(all_labels.max() + 1)], output_dict=True)
    for class_label, metrics in report.items():
        if class_label.isdigit():
            debug_message(f"Class {class_label} - Precision: {metrics['precision'] * 100:.2f}%, Recall: {metrics['recall'] * 100:.2f}%, F1-Score: {metrics['f1-score'] * 100:.2f}%")
    with open('test_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test F1 Score: {test_f1}\n")
        for class_label, metrics in report.items():
            if class_label.isdigit():
                f.write(f"Class {class_label} - Precision: {metrics['precision'] * 100:.2f}%, Recall: {metrics['recall'] * 100:.2f}%, F1-Score: {metrics['f1-score'] * 100:.2f}%\n")

        f.write(str(report))

    print(report)
    return test_acc, test_f1
