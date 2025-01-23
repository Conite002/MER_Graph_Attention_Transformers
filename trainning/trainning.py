
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

        debug_message(f"Epoch {epoch + 1} train loss", train_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(args['device'])
                labels = labels.to(args['device'])

                out = model(features, lengths=torch.tensor([len(features)]).to(args['device']))
                val_loss += criterion(out, labels).item()

        val_loss /= len(val_loader)
        debug_message(f"Epoch {epoch + 1} validation loss", val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            debug_message(f"Epoch {epoch + 1}: Best model saved")

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

    debug_message("Test Accuracy", test_acc)
    debug_message("Test F1 Score", test_f1)

    report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(all_labels.max() + 1)])
    with open('test_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc}\n")
        f.write(f"Test F1 Score: {test_f1}\n")
        f.write(report)

    print(report)
    return test_acc, test_f1
