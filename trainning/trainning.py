import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score
from tqdm import tqdm
from utils.debug import debug_message

def train_model(model, train_loader, val_loader, optimizer, criterion, args, epochs=10, save_model='best_model.pth', log_file='training_log.txt'):
    best_val_loss = float('inf')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    with open(log_file, 'w') as log:
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

            # Validation
            model.eval()
            val_loss = 0
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(args['device'])
                    labels = labels.to(args['device'])

                    out = model(features, lengths=torch.tensor([len(features)]).to(args['device']))
                    val_loss += criterion(out, labels).item()
                    preds = out.argmax(dim=1)
                    all_preds.append(preds.cpu())
                    all_labels.append(labels.cpu())

            val_loss /= len(val_loader)
            all_preds = torch.cat(all_preds)
            all_labels = torch.cat(all_labels)

            val_acc = accuracy_score(all_labels, all_preds)
            val_f1 = f1_score(all_labels, all_preds, average='macro')
            val_precision = precision_score(all_labels, all_preds, average='macro')
            val_recall = recall_score(all_labels, all_preds, average='macro')

            log.write(f"Epoch {epoch + 1} train loss {train_loss / len(train_loader):.4f}, val loss {val_loss:.4f}, val acc {val_acc * 100:.2f}%, val f1 {val_f1 * 100:.2f}%, val precision {val_precision * 100:.2f}%, val recall {val_recall * 100:.2f}%\n")
            debug_message(f"Epoch {epoch + 1} train loss {train_loss / len(train_loader):.4f}, val loss {val_loss:.4f}, val acc {val_acc * 100:.2f}%, val f1 {val_f1 * 100:.2f}%, val precision {val_precision * 100:.2f}%, val recall {val_recall * 100:.2f}%")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_model)
                log.write(f"Epoch {epoch + 1}: Best model saved\n")

def evaluate_model(model, test_loader, args, log_file):
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
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')

    # Write to the log file
    with open(log_file, 'a') as log:
        log.write(f"Test Accuracy: {test_acc * 100:.2f}%, Test F1 Score: {test_f1 * 100:.2f}%, Test Precision: {test_precision * 100:.2f}%, Test Recall: {test_recall * 100:.2f}%\n")
        debug_message(f"Test Accuracy: {test_acc * 100:.2f}%, Test F1 Score: {test_f1 * 100:.2f}%, Test Precision: {test_precision * 100:.2f}%, Test Recall: {test_recall * 100:.2f}%")

        report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(all_labels.max() + 1)], output_dict=True)
        for class_label, metrics in report.items():
            if class_label.isdigit():
                log.write(f"Class {class_label} - Precision: {metrics['precision'] * 100:.2f}%, Recall: {metrics['recall'] * 100:.2f}%, F1-Score: {metrics['f1-score'] * 100:.2f}%\n")
                debug_message(f"Class {class_label} - Precision: {metrics['precision'] * 100:.2f}%, Recall: {metrics['recall'] * 100:.2f}%, F1-Score: {metrics['f1-score'] * 100:.2f}%")

    with open('test_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test F1 Score: {test_f1}\n")
        f.write(f"Test Precision: {test_precision}\n")
        f.write(f"Test Recall: {test_recall}\n")
        for class_label, metrics in report.items():
            if class_label.isdigit():
                f.write(f"Class {class_label} - Precision: {metrics['precision'] * 100:.2f}%, Recall: {metrics['recall'] * 100:.2f}%, F1-Score: {metrics['f1-score'] * 100:.2f}%\n")

        f.write(str(report))

    print(report)
    return test_acc, test_f1, test_precision, test_recall
