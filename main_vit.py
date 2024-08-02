import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from vit_gridPE.vit import ViT, ViTRotate, ViTMerge, ViTComplex, ViTDeep

def load_data(data_dir, image_size, batch_size, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    return loader

def train_and_validate(model, train_loader, val_loader, epochs, lr):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
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
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
        
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
        print(f"Validation Accuracy after Epoch {epoch+1}: {accuracy:.4f}")
    
    print("Training and validation completed.")

def main_imagenet():
    image_size = 256
    patch_size = 32
    num_classes = 1000
    dim = 1024
    depth = 6
    heads = 16
    mlp_dim = 2048
    dropout = 0.1
    emb_dropout = 0.1

    # Initialize model
    v = ViTDeep(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout
    )

    # Load data
    train_loader = load_data("./img/train", image_size, batch_size=32)
    val_loader = load_data("./img/val", image_size, batch_size=32)

    # Train and validate model
    train_and_validate(v, train_loader, val_loader, epochs=10, lr=0.001)

def main_Caltech():
    image_size = 256
    patch_size = 32
    num_classes = 257
    dim = 1024
    depth = 6
    heads = 16
    mlp_dim = 2048
    dropout = 0.1
    emb_dropout = 0.1

    # Initialize model
    v = ViTDeep(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout
    )

    # Load data
    train_loader = load_data("./img/train", image_size, batch_size=32)
    val_loader = load_data("./img/val", image_size, batch_size=32)

    # Train and validate model
    train_and_validate(v, train_loader, val_loader, epochs=10, lr=0.001)
    
if __name__ == "__main__":
    main_Caltech()