"""Pre training before the attack."""
import torch
from torch import nn, optim

from leakpro.utils.logger import logger


def pre_train(model, train_loader, val_loader, epochs=60, lr=1e-4, device=None, patience=5):
    """Pre-train the model for face recognition with validation and early stopping."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.1)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        num_batches = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            predicted = (torch.sigmoid(outputs) >= 0.5).float()

            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.numel()
            num_batches += 1

        # Calculate training accuracy
        train_accuracy = correct_preds / total_preds * 100
        avg_train_loss = running_loss / num_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_preds = 0
        total_preds = 0
        num_batches = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)

                # Compute loss
                loss = criterion(outputs, labels.float())  # Ensure labels are float for BCEWithLogitsLoss
                val_loss += loss.item()

                # Apply threshold to get binary predictions
                predicted = (torch.sigmoid(outputs) >= 0.5).float()  # Sigmoid + thresholding

                # Calculate correct predictions
                correct_preds += (predicted == labels).sum().item()
                total_preds += labels.numel()  # Total number of attributes across all samples
                num_batches += 1

        # Compute validation metrics
        val_accuracy = correct_preds / total_preds * 100  # Percentage accuracy
        avg_val_loss = val_loss / num_batches

        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Logging the results
        logger.info(f"Epoch [{epoch+1}/{epochs}] - "
                    f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}% - "
                    f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        # Check early stopping condition
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # Reset counter if improvement is observed
            logger.info(f"Validation loss improved to {best_val_loss:.4f}.")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs!")
                break

        # Step the scheduler
        scheduler.step(avg_val_loss)

    logger.info("Training complete!")
