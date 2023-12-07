import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
import numpy as np
from initialize import class_labels
import torch.nn.functional as F

def plot_confusion_matrices(best_models, test_loader):
    for index, row in best_models.iterrows():
        model = row['model']
        model_name = row['model_name']  # Assuming you have a column 'model_name' in your DataFrame

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)
        # Set the model to evaluation mode
        model.eval()

        # Lists to store predictions, class probabilities, and ground truth labels
        all_predictions = []
        all_probabilities = []
        all_labels = []

        # Iterate over the test dataset
        i = 1
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                i = i + 1
                outputs = model(inputs)

                # Apply softmax to get class probabilities
                probabilities = F.softmax(outputs, dim=1)

                _, predictions = torch.max(outputs, 1)

                all_predictions.extend(predictions.numpy())
                all_probabilities.extend(probabilities.numpy())
                all_labels.extend(labels.numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        print(f"Model {model_name} - Test Accuracy: {accuracy:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_labels, yticklabels=class_labels)
        plt.title(f'Confusion Matrix - Model {model_name}')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

        safety_proba = [prob[0] for prob in all_probabilities]

        safety_true = [1 if i==0 else 0 for i in all_labels]

        fpr, tpr, thresholds = roc_curve(safety_true, safety_proba)
        roc_auc = auc(fpr, tpr)


        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.5f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

        baseline = safety_true.count(1)/ len(safety_true)

        precision, recall, thresholds_pr = precision_recall_curve(safety_true, safety_proba)
        pr_auc = auc(recall, precision)

        # Plot the Precision-Recall curve
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (area = {:.5f})'.format(pr_auc))
        plt.plot([0, 1], [baseline, baseline], linestyle='--', color='navy', label='Baseline = {:.5f}'.format(baseline))
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.show()