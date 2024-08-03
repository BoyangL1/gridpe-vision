import logging
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

# 设置日志记录器
def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 创建一个处理器，用于将日志写入文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建一个处理器，用于将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def train_and_validate(model, train_loader, val_loader, epochs, lr, accuracy_threshold=0.95, log_file="./log/vit_training.log"):
    logger = setup_logger(log_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    
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
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        logger.info(f"Validation Accuracy after Epoch {epoch+1}: {accuracy:.4f}")
        
        # Update learning rate and log it
        scheduler.step(accuracy)
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate after Epoch {epoch+1}: {current_lr:.6f}")
        
        # Check if accuracy threshold is reached
        if accuracy >= accuracy_threshold:
            logger.info(f"Accuracy threshold reached: {accuracy:.4f}, stopping training.")
            break
        
        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # torch.save(model.state_dict(), "best_model.pth")
            logger.info(f"Best model saved with accuracy: {accuracy:.4f}")
    
    logger.info("Training and validation completed.")