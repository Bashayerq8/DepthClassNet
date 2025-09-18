
# Copyright (c) 2025 Bashayer Abdallah
# Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
# Commercial use is prohibited.


import os, glob, torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torchvision.transforms as transforms


def test(model, dataloader, device, save_dir):
    model.eval()
    edge_detector = ScharrEdgeDetector()
    predictions, ground_truths = [], []
    class_predictions, class_ground_truths = [], []
    edge_predictions, edge_ground_truths = [], []
    ontology = model.sam_clip_encoder.ontology

    print("Starting testing...")
    with torch.no_grad():
        for rgb, depth, edge, class_labels in tqdm(dataloader, desc="Testing", leave=False):
            rgb, depth, edge = rgb.to(device), depth.to(device), edge.to(device)
            grayRGB = transforms.Grayscale(num_output_channels=1)(rgb)
            edge_rgb = edge_detector.forward(grayRGB)

            # Convert descriptive labels to indices
            target_indices = []
            for label in class_labels:
                try:
                    target_indices.append(ontology.index(label))
                except ValueError:
                    raise ValueError(f"Label '{label}' not in ontology: {ontology}")
            targets = torch.tensor(target_indices, dtype=torch.long, device=device)

            depth_pred, class_logits, probs, preds, prompts = model(rgb, edge_rgb, prompts=ontology)
            predicted_edges = edge_detector(depth_pred)

            predictions.append(depth_pred.cpu().numpy())
            ground_truths.append(depth.cpu().numpy())
            edge_predictions.append(predicted_edges.cpu().numpy())
            edge_ground_truths.append(edge.cpu().numpy())

            pred_classes = torch.argmax(class_logits, dim=1)
            class_predictions.extend(pred_classes.cpu().numpy())
            class_ground_truths.extend(targets.cpu().numpy())

    # Depth metrics
    depth_metrics = compute_errors(np.array(ground_truths), np.array(predictions))
    class_accuracy = accuracy_score(class_ground_truths, class_predictions)

    # Classification report
    unique_classes = np.unique(np.concatenate((class_ground_truths, class_predictions)))
    target_names = [f"Class {int(i)}" for i in unique_classes]
    class_report = classification_report(
        class_ground_truths, class_predictions,
        labels=unique_classes,
        target_names=target_names
    )

    print("Depth Error Metrics:")
    for k, v in depth_metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nClassification Metrics:")
    print(f"Accuracy: {class_accuracy:.4f}")
    print(class_report)

    # Save results
    os.makedirs(save_dir, exist_ok=True)
    test_metrics_file = os.path.join(save_dir, 'test_metrics.txt')
    with open(test_metrics_file, 'w') as f:
        f.write("Depth Metrics:\n")
        for k, v in depth_metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\nClassification Metrics:\n")
        f.write(f"Accuracy: {class_accuracy:.4f}\n")
        f.write(class_report)

    confusion = confusion_matrix(class_ground_truths, class_predictions)
    conf_matrix_file = os.path.join(save_dir, 'confusion_matrix.txt')
    np.savetxt(conf_matrix_file, confusion, fmt='%d', header='Confusion Matrix')
    print(f"Confusion Matrix saved to {conf_matrix_file}.")

    return predictions, ground_truths, class_predictions, class_ground_truths


####################################
# Main: Testing only
####################################
if __name__ == "__main__":
    batch_size = 1
    img_size = 224
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset (test only)
    root_dir = "Data/data"
    test_file = "Data/ucl_test.txt"
    _, _, test_dataset = prepare_dataset(root_dir, None, None, test_file)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Ontology from test set
    ontology = sorted(set(test_dataset.classes))
    print("Ontology:", ontology)

    # Load model + checkpoint
    model = DepthClass(ontology=ontology).to(device)
    checkpoint_path = "models_checkpoints/DepthClass/checkpoints/final_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {checkpoint_path}")

    # Run testing
    save_dir = os.path.dirname(checkpoint_path)
    test(model, test_loader, device, save_dir)

    print("Testing completed.")
