import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from ResNetClass import create_MDEclass
from Loss import BerHuLoss, SILogLoss, EdgeLoss
from classPrep import prepare_dataset
from Function import EarlyStoppingWithCheckpoint, compute_errors, ScharrEdgeDetector
from datetime import datetime
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# To suppress the warning messages
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

print(torch.cuda.is_available())


# Training function
def train(model, dataloader, criterion_depth, criterion_class, optimizer, device, scheduler=None):
    model.train()
    depth_alfa = 0.80
    class_alfa = 0.20
    total_depth_loss = 0.0
    total_class_loss = 0.0
    total_loss = 0.0
    total_a1 = 0.0  # Accuracy metric (a1)
    total_rmse = 0.0  # RMSE
    total_class_accuracy = 0.0  # Classification accuracy
    max_grad_norm = 1.0  # For gradient clipping

    print("Starting training...")
    for rgb, depth, edge, class_label in tqdm(dataloader, desc="Training", leave=False):
        rgb, depth, edge, class_label = rgb.to(device), depth.to(device), edge.to(device), class_label.to(device)

        optimizer.zero_grad()

        # Forward pass
        output, class_logits = model(rgb, class_label)

        # Calculate losses
        loss_depth = criterion_depth(output, depth)  # SILogLoss
        loss_class = criterion_class(class_logits, class_label)
        loss = (class_alfa * loss_class) + (depth_alfa * loss_depth)

        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            rmse = torch.sqrt(torch.mean((output - depth) ** 2)).item()
            thresh = torch.max(depth / output, output / depth)
            a1 = (thresh < 1.25).float().mean().item()
            pred_classes = torch.argmax(class_logits, dim=1)
            batch_class_accuracy = (pred_classes == class_label).float().mean().item()

        batch_size = rgb.size(0)
        total_depth_loss += loss_depth.item() * batch_size
        total_class_loss += loss_class.item() * batch_size
        total_loss += loss.item() * batch_size
        total_rmse += rmse * batch_size
        total_a1 += a1 * batch_size
        total_class_accuracy += batch_class_accuracy * batch_size

    if scheduler is not None:
        scheduler.step(total_loss / len(dataloader))

    avg_depth_loss = total_depth_loss / len(dataloader.dataset)
    avg_class_loss = total_class_loss / len(dataloader.dataset)
    avg_total_loss = total_loss / len(dataloader.dataset)
    avg_rmse = total_rmse / len(dataloader.dataset)
    avg_a1 = total_a1 / len(dataloader.dataset)
    avg_class_accuracy = total_class_accuracy / len(dataloader.dataset)

    print(f"Train - Depth Loss: {avg_depth_loss:.4f}, Class Loss: {avg_class_loss:.4f}, "
          f"Total Loss: {avg_total_loss:.4f}, RMSE: {avg_rmse:.4f}, Accuracy (a1): {avg_a1:.4f}, "
          f"Classification Accuracy: {avg_class_accuracy:.4f}")
    return avg_total_loss, avg_rmse, avg_a1, avg_class_accuracy


def validate(model, dataloader, criterion_depth, criterion_class, device):
    depth_alfa = 0.80
    class_alfa = 0.20
    total_depth_loss = 0.0
    total_class_loss = 0.0
    total_loss = 0.0
    total_a1 = 0.0
    total_rmse = 0.0
    total_class_accuracy = 0.0

    print("Starting validation...")
    with torch.no_grad():
        for rgb, depth, edge, class_label in tqdm(dataloader, desc="Validation", leave=False):
            rgb, depth, edge, class_label = rgb.to(device), depth.to(device), edge.to(device), class_label.to(device)

            output, class_logits = model(rgb, class_label)
            assert torch.all(output >= 0) and torch.all(output <= 1), "Depth output is not in [0, 1]!"
            assert class_logits.shape[
                       1] == num_classes, f"Expected {num_classes} classes, but got {class_logits.shape[1]}"

            loss_depth = criterion_depth(output, depth)
            loss_class = criterion_class(class_logits, class_label)
            loss = (class_alfa * loss_class) + (depth_alfa * loss_depth)

            rmse = torch.sqrt(torch.mean((output - depth) ** 2)).item()
            thresh = torch.max(depth / output, output / depth)
            a1 = (thresh < 1.25).float().mean().item()
            pred_classes = torch.argmax(class_logits, dim=1)
            batch_class_accuracy = (pred_classes == class_label).float().mean().item()

            batch_size = rgb.size(0)
            total_depth_loss += loss_depth.item() * batch_size
            total_class_loss += loss_class.item() * batch_size
            total_loss += loss.item() * batch_size
            total_rmse += rmse * batch_size
            total_a1 += a1 * batch_size
            total_class_accuracy += batch_class_accuracy * batch_size

    avg_depth_loss = total_depth_loss / len(dataloader.dataset)
    avg_class_loss = total_class_loss / len(dataloader.dataset)
    avg_total_loss = total_loss / len(dataloader.dataset)
    avg_rmse = total_rmse / len(dataloader.dataset)
    avg_a1 = total_a1 / len(dataloader.dataset)
    avg_class_accuracy = total_class_accuracy / len(dataloader.dataset)

    print(f"Validation - Depth Loss: {avg_depth_loss:.4f}, Class Loss: {avg_class_loss:.4f}, "
          f"Total Loss: {avg_total_loss:.4f}, RMSE: {avg_rmse:.4f}, Accuracy (a1): {avg_a1:.4f}, "
          f"Classification Accuracy: {avg_class_accuracy:.4f}")
    return avg_total_loss, avg_rmse, avg_a1, avg_class_accuracy


def test(model, dataloader, device):
    model.eval()
    predictions, ground_truths = [], []
    class_predictions, class_ground_truths = [], []

    print("Starting testing...")
    with torch.no_grad():
        for rgb, depth, edge, class_label in tqdm(dataloader, desc="Testing", leave=False):
            rgb, depth, edge, class_label = rgb.to(device), depth.to(device), edge.to(device), class_label.to(device)

            output, class_logits = model(rgb, class_label)

            predictions.append(output.cpu().numpy())
            ground_truths.append(depth.cpu().numpy())
            pred_classes = torch.argmax(class_logits, dim=1)
            class_predictions.extend(pred_classes.cpu().numpy())
            class_ground_truths.extend(class_label.cpu().numpy())

    depth_metrics = compute_errors(np.array(ground_truths), np.array(predictions))
    class_accuracy = accuracy_score(class_ground_truths, class_predictions)
    class_report = classification_report(class_ground_truths, class_predictions, target_names=[f"Class {i}" for i in range(len(set(class_ground_truths)))])

    print("Depth Error Metrics:")
    for key, value in depth_metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Metrics:")
    print(f"Accuracy: {class_accuracy:.4f}")
    print(class_report)

    test_metrics_file = os.path.join(args.save_dir, 'test_metrics.txt')
    with open(test_metrics_file, 'w') as f:
        f.write("Depth Metrics:\n")
        for key, value in depth_metrics.items():
            f.write(f"{key}: {value:.4f}\n")
        f.write("\nClassification Metrics:\n")
        f.write(f"Accuracy: {class_accuracy:.4f}\n")
        f.write(class_report)

    confusion = confusion_matrix(class_ground_truths, class_predictions)
    conf_matrix_file = os.path.join(args.save_dir, 'confusion_matrix.txt')
    np.savetxt(conf_matrix_file, confusion, fmt='%d', header='Confusion Matrix')
    print(f"Confusion Matrix saved to {conf_matrix_file}.")

    return predictions, ground_truths, class_predictions, class_ground_truths


if __name__ == "__main__":
    # Hyperparameters
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 50
    num_classes = 9
    img_size = 256

    import torch

    # print(torch.cuda.is_available())  # Should return True
    # print(torch.cuda.device_count())  # Number of available GPUs
    # print(torch.cuda.get_device_name(0))  # Name of the first GPU

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    root_dir = "Data/data"
    train_file = "Data/ucl_train.txt"
    val_file = "Data/ucl_val.txt"
    test_file = "Data/ucl_test.txt"

    train_dataset, val_dataset, test_dataset = prepare_dataset(root_dir, train_file, val_file, test_file)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = create_MDEclass(num_classes=num_classes, img_size=img_size).to(device)
    criterion_depth = SILogLoss()
    criterion_class = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    model_name = model.__class__.__name__
    model_folder = os.path.join('models_checkpoints', model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    metrics_file = os.path.join(model_folder, 'training_validation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("Epoch\tTrain_Loss\tTrain_RMSE\tTrain_a1\tVal_Loss\tVal_RMSE\tVal_a1\tTrain_Class_Acc\tVal_Class_Acc\n")

    checkpoint_path = os.path.join(model_folder, 'checkpoint_epoch_*.pth')
    try:
        checkpoint = torch.load(os.path.join(model_folder, 'epoch_3_2024-12-09.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print("Loaded the best model and optimizer from checkpoint for training.")
    except FileNotFoundError:
        start_epoch = 0
        print("No checkpoint found, starting training from scratch.")

    early_stopping = EarlyStoppingWithCheckpoint(patience=5, min_delta=0.01, checkpoint_path=checkpoint_path)

    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}] starting...")

        train_loss, train_rmse, train_a1, train_class_accuracy = train(
            model, train_loader, criterion_depth, criterion_class, optimizer, device, scheduler=scheduler
        )

        val_loss, val_rmse, val_a1, val_class_accuracy = validate(
            model, val_loader, criterion_depth, criterion_class, device
        )

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train RMSE: {train_rmse:.4f}, "
              f"Train a1: {train_a1:.4f}, Train Class Acc: {train_class_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val RMSE: {val_rmse:.4f}, Val a1: {val_a1:.4f}, "
              f"Val Class Acc: {val_class_accuracy:.4f}")

        with open(metrics_file, 'a') as f:
            f.write(f"{epoch + 1}\t{train_loss:.4f}\t{train_rmse:.4f}\t{train_a1:.4f}\t"
                    f"{val_loss:.4f}\t{val_rmse:.4f}\t{val_a1:.4f}\t{train_class_accuracy:.4f}\t{val_class_accuracy:.4f}\n")

        current_date = datetime.now().strftime('%Y-%m-%d')
        epoch_checkpoint_path = os.path.join(model_folder, f'epoch_{epoch + 1}_{current_date}.pth')
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

        if early_stopping(val_loss, model):
            print("Early stopping")
            break

    final_model_path = os.path.join(model_folder, 'epoch_3_2024-12-09.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }, final_model_path)
    print(f"Final model saved as {final_model_path}")

    checkpoint = torch.load(final_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded the final fine-tuned model for testing.")

    predictions, ground_truths, class_predictions, class_ground_truths = test(model, test_loader, device)
    print("Testing completed.")
