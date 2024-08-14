import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def train_and_validate(model, train_loader, val_loader, epochs, lr, accuracy_threshold=0.95, log_file="./log/vit_training.log"):
    logger = setup_logger(log_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)
    
    best_accuracy = 0.0
    
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted_top1 = torch.max(outputs.data, 1)
                _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)
                
                total += labels.size(0)
                correct_top1 += (predicted_top1 == labels).sum().item()
                correct_top5 += sum([labels[i] in predicted_top5[i] for i in range(labels.size(0))])
        
        accuracy_top1 = correct_top1 / total
        accuracy_top5 = correct_top5 / total
        
        logger.info(f"Validation Accuracy after Epoch {epoch+1} (Top-1): {accuracy_top1:.4f}")
        logger.info(f"Validation Accuracy after Epoch {epoch+1} (Top-5): {accuracy_top5:.4f}")
        
        # Update learning rate and log it
        scheduler.step(accuracy_top1)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate after Epoch {epoch+1}: {current_lr:.6f}")
        
        # Check if accuracy threshold is reached
        if accuracy_top1 >= accuracy_threshold:
            logger.info(f"Accuracy threshold reached: {accuracy_top1:.4f}, stopping training.")
            break
        
        # Save the best model
        if accuracy_top1 > best_accuracy:
            best_accuracy = accuracy_top1
            # torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"Best model saved with accuracy: {accuracy_top1:.4f}")
    
    logger.info("Training and validation completed.")

def load_data(data_dir, image_size, batch_size, val_split=0.2, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    
    # Split the dataset into train and validation sets
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader