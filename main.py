import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import medmnist
from medmnist import INFO, Evaluator
from torch.utils.data import random_split
from torchvision import models
from tqdm import tqdm
import copy
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin
from PIL import Image
import requests

if __name__ == "__main__":
    try:
        data_flag = "tissuemnist"
        download = True

        BATCH_SIZE = 128
        lr = 0.001

        info = INFO[data_flag]
        task = info["task"]
        n_channels = info["n_channels"]
        n_classes = len(info["label"])

        DataClass = getattr(medmnist, info["python_class"])

        # preprocessing
        data_transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

        # load the data
        train_dataset = DataClass(
            split="train", transform=data_transform, download=download, size=224
        )
        test_dataset = DataClass(
            split="test", transform=data_transform, download=download, size=224
        )
        cal_dataset = DataClass(
            split="val", transform=data_transform, download=download, size=224
        )

        # split training set into train and validation
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

        # encapsulate data into dataloader form
        train_loader = data.DataLoader(
            dataset=train_subset, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = data.DataLoader(
            dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False
        )
        val_loader = data.DataLoader(
            dataset=val_subset, batch_size=2 * BATCH_SIZE, shuffle=False
        )

        print(train_dataset)
        print("===================")
        print(test_dataset)

        device = "cuda"

        class DenseNetCustom(nn.Module):
            def __init__(self, num_classes=1000):
                super(DenseNetCustom, self).__init__()
                self.densenet = models.densenet121(pretrained=True)
                self.densenet.classifier = nn.Linear(1024, num_classes)

            def forward(self, x):
                x = self.densenet(x)
                return x

        model = DenseNetCustom(num_classes=n_classes)
        model.to(device)
        print(model)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        num_epochs = 8
        train_losses, val_losses = [], []
        train_acc_list, val_acc_list = [], []
        train_auc_list, val_auc_list = [], []
        best_val_acc = 0.0
        best_model_weights = None
        best_epoch = 0

        for epoch in range(num_epochs):
            # Training
            model.train()
            running_loss = 0.0
            correct, total = 0, 0
            train_probs, train_labels_all = [], []
            print(f"Started Epoch {epoch+1}")
            for inputs, labels in tqdm(train_loader):
                inputs, labels = inputs.to(device), labels.to(device).squeeze()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                train_probs.append(F.softmax(outputs, dim=1).detach().cpu().numpy())
                train_labels_all.append(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = 100.0 * correct / total
            train_losses.append(train_loss)
            train_acc_list.append(train_acc)
            train_probs = np.concatenate(train_probs)
            train_labels_all = np.concatenate(train_labels_all)
            train_auc = roc_auc_score(train_labels_all, train_probs, multi_class="ovr")
            train_auc_list.append(train_auc)

            # Validation
            model.eval()
            val_running_loss = 0.0
            correct, total = 0, 0
            val_probs, val_labels_all = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device).squeeze()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                    val_probs.append(F.softmax(outputs, dim=1).cpu().numpy())
                    val_labels_all.append(labels.cpu().numpy())

            val_loss = val_running_loss / len(val_loader.dataset)
            val_acc = 100.0 * correct / total
            val_losses.append(val_loss)
            val_acc_list.append(val_acc)
            val_probs = np.concatenate(val_probs)
            val_labels_all = np.concatenate(val_labels_all)
            val_auc = roc_auc_score(val_labels_all, val_probs, multi_class="ovr")
            val_auc_list.append(val_auc)

            # Save best weights
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_weights = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1

            scheduler.step()
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Train AUC: {train_auc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}"
            )

        # Load best weights
        model.load_state_dict(best_model_weights)
        print(
            f"\nLoaded best model weights from Epoch {best_epoch} with Val Acc: {best_val_acc:.2f}%"
        )

        save_path = f"densenet_tissuemnist_epoch{best_epoch}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        class TorchSklearnWrapper(BaseEstimator, ClassifierMixin):
            """Wraps a PyTorch classifier to be compatible with MAPIE/sklearn."""

            def __init__(
                self, torch_model, device="cuda", transform=None, batch_size=256
            ):
                self.torch_model = torch_model
                self.device = device
                self.transform = transform
                self.batch_size = batch_size
                self.classes_ = None

            def fit(self, X, y=None):
                return self

            def predict(self, X):
                proba = self.predict_proba(X)
                return self.classes_[np.argmax(proba, axis=1)]

            def predict_proba(self, X):
                self.torch_model.eval()
                all_proba = []
                dataset = torch.utils.data.TensorDataset(torch.tensor(X))
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=False
                )
                with torch.no_grad():
                    for (batch,) in loader:
                        if self.transform is not None:
                            imgs = []
                            for img in batch.numpy():
                                img = Image.fromarray(img.astype(np.uint8))
                                imgs.append(self.transform(img))
                            batch = torch.stack(imgs)
                        else:
                            batch = batch.float()
                        batch = batch.to(self.device)
                        logits = self.torch_model(batch)
                        proba = F.softmax(logits, dim=1)
                        all_proba.append(proba.cpu().numpy())
                return np.vstack(all_proba)

        wrapped_model = TorchSklearnWrapper(
            model, device=device, transform=data_transform
        )

        wrapped_model.fit(cal_dataset.imgs, cal_dataset.labels.ravel())

        # Conformal prediction
        from mapie.classification import SplitConformalClassifier

        confidence_level = 0.95
        mapie_classifier = SplitConformalClassifier(
            estimator=wrapped_model, confidence_level=confidence_level, prefit=True
        )
        mapie_classifier.conformalize(cal_dataset.imgs, cal_dataset.labels)
        y_pred, y_pred_set = mapie_classifier.predict_set(test_dataset.imgs)

        import pandas as pd
        import json

        label_map = info["label"]

        # --- Save conformal prediction results (per test sample) ---
        records = []
        for i in range(len(y_pred)):
            pred_label = int(y_pred[i])
            true_label = (
                int(test_dataset.labels[i].item())
                if hasattr(test_dataset.labels[i], "item")
                else int(test_dataset.labels[i])
            )
            # y_pred_set shape: (n_samples, n_classes) — boolean mask
            prediction_set = [int(c) for c in range(n_classes) if y_pred_set[i, c]]
            prediction_set_names = [label_map[str(c)] for c in prediction_set]
            records.append(
                {
                    "sample_index": i,
                    "true_label": true_label,
                    "true_label_name": label_map[str(true_label)],
                    "predicted_label": pred_label,
                    "predicted_label_name": label_map[str(pred_label)],
                    "prediction_set": prediction_set,
                    "prediction_set_names": prediction_set_names,
                    "set_size": len(prediction_set),
                    "true_label_in_set": true_label in prediction_set,
                }
            )

        df_conformal = pd.DataFrame(records)
        df_conformal.to_csv("conformal_prediction_results.csv", index=False)
        print(f"Saved conformal prediction results to conformal_prediction_results.csv")

        # --- Save training metrics ---
        df_training = pd.DataFrame(
            {
                "epoch": list(range(1, len(train_losses) + 1)),
                "train_loss": train_losses,
                "train_accuracy": train_acc_list,
                "test_accuracy": test_acc_list,
            }
        )
        df_training.to_csv("training_metrics.csv", index=False)
        print(f"Saved training metrics to training_metrics.csv")

        # --- Print summary statistics ---
        coverage = df_conformal["true_label_in_set"].mean()
        avg_set_size = df_conformal["set_size"].mean()
        print(f"\nConformal Prediction Summary (confidence={confidence_level}):")
        print(f"  Coverage (empirical): {coverage:.4f}")
        print(f"  Average set size:     {avg_set_size:.4f}")
        print(f"\nSet size distribution:")
        print(df_conformal["set_size"].value_counts().sort_index().to_string())

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.plot(train_acc_list, label="Train Accuracy")
        plt.plot(val_acc_list, label="Val Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy (%)")
        plt.title("Accuracy Curve")
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.plot(train_auc_list, label="Train AUC")
        plt.plot(val_auc_list, label="Val AUC")
        plt.xlabel("Epochs")
        plt.ylabel("AUC")
        plt.title("AUC Curve")
        plt.legend()

        plt.tight_layout()
        plt.show()
    finally:
        pod_id = ""
        token = ""

        url = f"https://rest.runpod.io/v1/pods/{pod_id}/stop"
        headers = {"Authorization": f"Bearer {token}"}

        response = requests.post(url, headers=headers)
        print(response.status_code, response.json())
