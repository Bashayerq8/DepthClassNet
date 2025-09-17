
'''

Copyright (c) 2025 Bashayer Abdallah
Licensed under CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
Commercial use is prohibited.

'''
import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from Function import ScharrEdgeDetector, compute_errors, EarlyStoppingWithCheckpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau  
import torchvision.transforms as transforms
from autodistill_sam_clip import SAMCLIP  # Import SAMCLIP from autodistill_sam_clip
from Loss import SILogLoss, EdgeLoss, depth_loss_function
from classPrep import prepare_dataset
from DepthClass import DepthClass



import warnings

warnings.filterwarnings("ignore", category=UserWarning)

torch.cuda.empty_cache()


####################################
# Training Function
####################################

def train(model, dataloader, criterion_depth, criterion_class, criterion_edge,
          optimizer, device, scheduler=None):
    model.train()
    # Loss weights
    depth_alfa = 0.60
    class_alfa = 0.30
    edge_alfa = 0.10

    total_depth_loss = 0.0
    total_class_loss = 0.0
    total_edge_loss = 0.0
    total_loss = 0.0
    total_a1 = 0.0  # Depth accuracy metric (a1)
    total_rmse = 0.0  # RMSE
    total_class_accuracy = 0.0  # Classification accuracy
    max_grad_norm = 1.0  # For gradient clipping

    edge_detector = ScharrEdgeDetector()

    # Use the model's internal ontology to map descriptive labels to indices.
    ontology = (model.module.sam_clip_encoder.ontology
                if hasattr(model, "module")
                else model.sam_clip_encoder.ontology)

    # print("Ontology (classes):", ontology)

    print("Starting training...")

    for rgb, depth, edge, class_labels in tqdm(dataloader, desc="Training", leave=False):
        rgb, depth, edge = rgb.to(device), depth.to(device), edge.to(device)

        # Convert descriptive labels to integer targets.
        # initializes an empty list target_indices to store numeric class indices.
        target_indices = []
        for label in class_labels:
            try:
                target_indices.append(ontology.index(label))
            except ValueError:
                raise ValueError(f"Label '{label}' not found in the ontology: {ontology}")
        # print("target_indices:", target_indices)

        targets = torch.tensor(target_indices, dtype=torch.long, device=device)

        # print("targets:", targets)

        optimizer.zero_grad()

        # Forward pass.
        depth_pred, class_logits, probs, preds, prompts = model(rgb, edge, prompts=ontology)
        predicted_edges = edge_detector(depth_pred)

        # Compute losses.
        loss_depth = criterion_depth(depth_pred, depth)
        loss_class = criterion_class(class_logits, targets)
        loss_edge = criterion_edge(predicted_edges, edge)
        loss = (depth_alfa * loss_depth) + (class_alfa * loss_class) + (edge_alfa * loss_edge)
        loss.backward()

        # Gradient clipping and optimizer step.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            rmse = torch.sqrt(torch.mean((depth_pred - depth) ** 2)).item()
            thresh = torch.max(depth / depth_pred, depth_pred / depth)
            a1 = (thresh < 1.25).float().mean().item()
            pred_classes = torch.argmax(class_logits, dim=1)
            batch_class_accuracy = (pred_classes == targets).float().mean().item()

        # Get batch size.
        batch_size = rgb.size(0)
        # Accumulate losses weighted by batch size.
        total_depth_loss += loss_depth.item() * batch_size
        total_edge_loss += loss_edge.item() * batch_size
        total_loss += loss.item() * batch_size
        total_rmse += rmse * batch_size
        total_a1 += a1 * batch_size
        total_class_loss += loss_class.item() * batch_size
        total_class_accuracy += batch_class_accuracy * batch_size

    # Step scheduler per batch (if using a per-batch scheduler).
    if scheduler is not None:
        scheduler.step(total_loss / len(dataloader))

    # Compute averages over the entire dataset.
    dataset_size = len(dataloader.dataset)
    avg_depth_loss = total_depth_loss / dataset_size
    avg_class_loss = total_class_loss / dataset_size
    avg_edge_loss = total_edge_loss / dataset_size
    avg_total_loss = total_loss / dataset_size
    avg_rmse = total_rmse / dataset_size
    avg_a1 = total_a1 / dataset_size
    avg_class_accuracy = total_class_accuracy / dataset_size

    print(f"Train - Depth Loss: {avg_depth_loss:.4f}, Class Loss: {avg_class_loss:.4f}, "
          f"Edge Loss: {avg_edge_loss:.4f}, Total Loss: {avg_total_loss:.4f}, "
          f"RMSE: {avg_rmse:.4f}, a1: {avg_a1:.4f}, Classification Accuracy: {avg_class_accuracy:.4f}")

    return avg_total_loss, avg_rmse, avg_a1, avg_class_accuracy


####################################
# Validation Function
####################################
def validate(model, dataloader, criterion_depth, criterion_class, criterion_edge, device):
    depth_alfa = 0.60
    class_alfa = 0.30
    edge_alfa = 0.10
    edge_detector = ScharrEdgeDetector()

    total_depth_loss = 0.0
    total_class_loss = 0.0
    total_edge_loss = 0.0
    total_loss = 0.0
    total_a1 = 0.0
    total_rmse = 0.0
    total_class_accuracy = 0.0

    ontology = model.sam_clip_encoder.ontology
    num_classes = len(ontology)
    # print("Ontology (of validation):", ontology)

    print("Starting validation...")
    model.eval()
    with torch.no_grad():
        for rgb, depth, edge, class_labels in tqdm(dataloader, desc="Validation", leave=False):
            rgb, depth, edge = rgb.to(device), depth.to(device), edge.to(device)
            greyScale_trans = transforms.Grayscale(num_output_channels=1)
            grayRGB = greyScale_trans(rgb)
            edge_rgb = edge_detector.forward(grayRGB)

            # Convert descriptive labels to indices.
            target_indices = []
            for label in class_labels:
                try:
                    target_indices.append(ontology.index(label))
                except ValueError:
                    raise ValueError(f"Label '{label}' not found in the ontology: {ontology}")
            targets = torch.tensor(target_indices, dtype=torch.long, device=device)

            depth_pred, class_logits, probs, preds, prompts = model(rgb, edge_rgb, prompts=ontology)

            predicted_edges = edge_detector(depth_pred)
            # Sanity checks (adjust as needed).
            assert torch.all(depth_pred >= 0) and torch.all(depth_pred <= 1), "Depth output is not in [0, 1]!"
            assert class_logits.shape[
                       1] == num_classes, f"Expected {num_classes} classes, but got {class_logits.shape[1]}"

            loss_depth = criterion_depth(depth_pred, depth)
            loss_class = criterion_class(class_logits, targets)
            loss_edge = criterion_edge(predicted_edges, edge)
            loss = (depth_alfa * loss_depth) + (class_alfa * loss_class) + (edge_alfa * loss_edge)

            rmse = torch.sqrt(torch.mean((depth_pred - depth) ** 2)).item()
            thresh = torch.max(depth / depth_pred, depth_pred / depth)
            a1 = (thresh < 1.25).float().mean().item()
            pred_classes = torch.argmax(class_logits, dim=1)
            batch_class_accuracy = (pred_classes == targets).float().mean().item()

            # print("Predicted classes:", pred_classes.cpu().numpy())
            # print("Target indices:", targets.cpu().numpy())
            # print("Ontology:", ontology)
            # print("Predicted logits:", class_logits[:5].detach().cpu().numpy())
            # print("Predicted classes:", torch.argmax(class_logits, dim=1)[:5].detach().cpu().numpy())
            # print("Targets:", targets[:5].detach().cpu().numpy())

            batch_size = rgb.size(0)
            total_depth_loss += loss_depth.item() * batch_size
            total_class_loss += loss_class.item() * batch_size
            total_edge_loss += loss_edge.item() * batch_size
            total_loss += loss.item() * batch_size
            total_rmse += rmse * batch_size
            total_a1 += a1 * batch_size
            total_class_accuracy += batch_class_accuracy * batch_size

    avg_depth_loss = total_depth_loss / len(dataloader.dataset)
    avg_class_loss = total_class_loss / len(dataloader.dataset)
    avg_edge_loss = total_edge_loss / len(dataloader.dataset)
    avg_total_loss = total_loss / len(dataloader.dataset)
    avg_rmse = total_rmse / len(dataloader.dataset)
    avg_a1 = total_a1 / len(dataloader.dataset)
    avg_class_accuracy = total_class_accuracy / len(dataloader.dataset)

    print(f"Validation - Depth Loss: {avg_depth_loss:.4f}, Class Loss: {avg_class_loss:.4f}, "
          f"Edge Loss: {avg_edge_loss:.4f}, Total Loss: {avg_total_loss:.4f}, "
          f"RMSE: {avg_rmse:.4f}, a1: {avg_a1:.4f}, Classification Accuracy: {avg_class_accuracy:.4f}")
    return avg_total_loss, avg_rmse, avg_a1, avg_class_accuracy


####################################
# Test Function
####################################
def test(model, dataloader, device, save_dir):
    model.eval()
    edge_detector = ScharrEdgeDetector()
    predictions, ground_truths = [], []
    class_predictions, class_ground_truths = [], []
    # Optionally, you can also collect predicted edges.
    edge_predictions = []
    edge_ground_truths = []
    ontology = model.sam_clip_encoder.ontology

    print("Starting testing...")
    with torch.no_grad():
        for rgb, depth, edge, class_labels in tqdm(dataloader, desc="Testing", leave=False):
            rgb, depth, edge = rgb.to(device), depth.to(device), edge.to(device)
            greyScale_trans = transforms.Grayscale(num_output_channels=1)
            grayRGB = greyScale_trans(rgb)
            edge_rgb = edge_detector.forward(grayRGB)

            # Convert descriptive labels to indices.
            target_indices = []
            for label in class_labels:
                try:
                    target_indices.append(ontology.index(label))
                except ValueError:
                    raise ValueError(f"Label '{label}' not found in the ontology: {ontology}")
            targets = torch.tensor(target_indices, dtype=torch.long, device=device)

            depth_pred, class_logits, probs, preds, prompts = model(rgb, edge_rgb, prompts=ontology)
            predicted_edges = edge_detector(depth_pred)

            predictions.append(depth_pred.cpu().numpy())
            ground_truths.append(depth.cpu().numpy())
            # Compute and store edge predictions.
            edge_predictions.append(predicted_edges.cpu().numpy())
            edge_ground_truths.append(edge.cpu().numpy())

            pred_classes = torch.argmax(class_logits, dim=1)
            class_predictions.extend(pred_classes.cpu().numpy())
            class_ground_truths.extend(targets.cpu().numpy())
            import numpy as np
            print("Unique classes in class_ground_truths:", np.unique(class_ground_truths))
            print("Unique classes in class_predictions:", np.unique(class_predictions))

    # Compute depth error metrics (assumes you have defined compute_errors).
    depth_metrics = compute_errors(np.array(ground_truths), np.array(predictions))
    class_accuracy = accuracy_score(class_ground_truths, class_predictions)

    # Combine both true and predicted labels to ensure you capture all unique classes
    unique_classes = np.unique(np.concatenate((class_ground_truths, class_predictions)))
    target_names = [f"Class {int(i)}" for i in unique_classes]

    class_report = classification_report(
        class_ground_truths, class_predictions,
        labels=unique_classes,  # explicitly specify which labels to use
        target_names=target_names
    )

    print("Depth Error Metrics:")
    for key, value in depth_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Metrics:")
    print(f"Accuracy: {class_accuracy:.4f}")
    print(class_report)

    test_metrics_file = os.path.join(save_dir, 'test_metrics.txt')
    with open(test_metrics_file, 'w') as f:
        f.write("Depth Metrics:\n")
        for key, value in depth_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nClassification Metrics:\n")
        f.write(f"Accuracy: {class_accuracy:.4f}\n")
        f.write(class_report)

    confusion = confusion_matrix(class_ground_truths, class_predictions)
    conf_matrix_file = os.path.join(save_dir, 'confusion_matrix.txt')
    np.savetxt(conf_matrix_file, confusion, fmt='%d', header='Confusion Matrix')
    print(f"Confusion Matrix saved to {conf_matrix_file}.")

    return (predictions, ground_truths, class_predictions, class_ground_truths)


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 6
    learning_rate = 0.000357
    num_epochs = 50
    num_classes = 6
    img_size = 224

    # GPU setup
    print(torch.cuda.device_count())  # Number of available GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset setup
    root_dir = "Data/data"

    train_dataset, val_dataset, test_dataset = prepare_dataset(root_dir)

    train_loader = DataLoader(dataset=train_dataset, batch_size=6, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=6, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    train_labels = [train_dataset.dataset.classes[i] for i in train_dataset.indices]
    val_labels = [val_dataset.dataset.classes[i] for i in val_dataset.indices]
    test_labels = [test_dataset.dataset.classes[i] for i in test_dataset.indices]
    all_labels = set(train_labels + val_labels + test_labels)
    ontology = sorted(set(all_labels))

    # train_file = "Data/ucl_train.txt"
    # val_file = "Data/ucl_val.txt"
    # test_file = "Data/ucl_test.txt"
    # train_dataset, val_dataset, test_dataset = prepare_dataset(root_dir, train_file, val_file, test_file)
    #
    # # DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Build ontology from dataset labels
    # all_labels = set(train_dataset.classes) | set(val_dataset.classes) | set(test_dataset.classes)
    # ontology = sorted(set(all_labels))
    # print("Ontology (union of classes):", ontology)

    ################## to use GPU by id ##################
    # Create model
    model = DepthClass(ontology=ontology).to(device)
    # model = DepthClass_concat(ontology=ontology).to(device)

    # if torch.cuda.device_count() > 1:
    #     print("Using multiple GPUs...")
    #     model = nn.DataParallel(model)

    # Loss functions
    criterion_depth = SILogLoss()
    criterion_edge = EdgeLoss()
    criterion_class = nn.CrossEntropyLoss()
    depth_nLoss = nn.SmoothL1Loss()  # Huber Loss for robust training

    # ============================
    # 1. Set up directories
    # ============================
    model_name = model.__class__.__name__
    model_folder = os.path.join('models_checkpoints', model_name)
    os.makedirs(model_folder, exist_ok=True)

    checkpoint_dir = os.path.join(model_folder, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    metrics_file = os.path.join(checkpoint_dir, 'training_validation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\tTrain_RMSE\tTrain_a1\tVal_Loss\tVal_RMSE\tVal_a1\tTrain_Class_Acc\tVal_Class_Acc\n")

    # ============================
    # 2. Load the latest checkpoint (if any)
    # ============================
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    start_epoch = 0
    optimizer_state = None  # placeholder for optimizer state

    if checkpoint_files:
        # Select the checkpoint with the latest modification time
        latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
        checkpoint = torch.load(latest_checkpoint)
        # Compare the keys of the checkpoint with the current model's keys
        checkpoint_keys = set(checkpoint['model_state_dict'].keys())
        model_keys = set(model.state_dict().keys())

        # If the checkpoint keys include all model keys, load the checkpoint.
        if model_keys.issubset(checkpoint_keys):
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer_state = checkpoint.get('optimizer_state_dict', None)
            start_epoch = checkpoint['epoch'] + 1  # next epoch to run
            print(f"Loaded checkpoint from {latest_checkpoint}. Starting from epoch {start_epoch}.")
        else:
            print("Warning: Checkpoint keys do not match the current model architecture. Skipping checkpoint loading.")

    # ============================
    # 3. Create optimizer, early stopping, and scheduler
    # ============================
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Restore the optimizer state if available.
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    # Initialize early stopping unconditionally.
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
    # early_stopping = EarlyStoppingWithCheckpoint(patience=5, min_delta=0.001, checkpoint_path=best_checkpoint_path)

    # Use ReduceLROnPlateau as the scheduler.
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # ============================
    # 4. Training and validation loop
    # ============================
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}] starting...")

        # Training: note that criterion_edge is now required.
        train_loss, train_rmse, train_a1, train_class_accuracy = train(
            model,
            train_loader,
            criterion_depth,
            criterion_class,
            criterion_edge,  # edge loss function
            optimizer,
            device,
            scheduler=None  # remove per-batch scheduler stepping
        )

        # Validation also requires the edge loss criterion.
        val_loss, val_rmse, val_a1, val_class_accuracy = validate(
            model,
            val_loader,
            criterion_depth,
            criterion_class,
            criterion_edge,
            device
        )

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
              f"Train a1: {train_a1:.4f}, Train Class Acc: {train_class_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, "
              f"Val a1: {val_a1:.4f}, Val Class Acc: {val_class_accuracy:.4f}")

        # Append training and validation metrics to the file after each epoch.
        with open(metrics_file, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_rmse:.4f}\t{train_a1:.4f}\t"
                    f"{val_loss:.4f}\t{val_rmse:.4f}\t{val_a1:.4f}\t{train_class_accuracy:.4f}\t{val_class_accuracy:.4f}\n")

        # Step the learning rate scheduler on validation loss
        scheduler.step(val_loss)
        # Save a checkpoint for the current epoch
        current_date = datetime.now().strftime('%Y-%m-%d')
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}_{current_date}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_rmse': train_rmse,
            'train_a1': train_a1,
            'train_class_accuracy': train_class_accuracy,
            'val_loss': val_loss,
            'val_rmse': val_rmse,
            'val_a1': val_a1,
            'val_class_accuracy': val_class_accuracy
        }, epoch_checkpoint_path)
        print(f"Checkpoint for epoch {epoch + 1} saved at {epoch_checkpoint_path}.")

        Check early stopping criteria
        if early_stopping(val_loss, model):
            print("Early stopping triggered.")
            break

    # Save the final model checkpoint (using a fixed filename)
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, final_model_path)
    print(f"Final model saved as {final_model_path}")

    # Load the final fine-tuned model for testing.
    checkpoint = torch.load(final_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded the final fine-tuned model for testing.")

    predictions, ground_truths, class_predictions, class_ground_truths = test(model, test_loader, device,
                                                                              checkpoint_dir)

    print("Testing completed.")


