import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models

def arg_parse():
    parser = argparse.ArgumentParser(description='Image Classifier Training')
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default="checkpoint.pth", help='Directory to save checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=150, help='Number of hidden units')
    parser.add_argument('--output_features', type=int, default=102, help='Number of output features')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--arch', default='resnet18', help='Choose architecture')
    return parser.parse_args()

def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

def get_loader(data_dir, transform, batch_size=64, shuffle=True):
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def check_device(use_gpu):
    return torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")

def load_model(arch):
    model = getattr(models, arch)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    return model

def initialize_classifier(model, hidden_units, output_features):
    in_features = model.fc.in_features if hasattr(model, 'fc') else model.classifier.in_features
    return nn.Sequential(
        nn.Linear(in_features, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, output_features),
        nn.LogSoftmax(dim=1)
    )

def train_model(model, trainloader, validloader, device, optimizer, criterion, epochs, print_every):
    model.to(device)
    steps = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                test_loss, correct = 0, 0
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        output = model(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()
                        ps = torch.exp(output)
                        _, top_class_idx = ps.topk(1, dim=1)
                        correct += torch.mean((top_class_idx == labels.view(*top_class_idx.shape)).float()).item()
                print(f"Epoch {epoch+1}/{epochs}.. Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {correct/len(validloader):.3f}")
                running_loss = 0
    return model

def save_checkpoint(model, path, arch, hidden_units, output_features, class_to_idx):
    checkpoint = {
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'arch': arch,
        'hidden_units': hidden_units,
        'output_features': output_features
    }
    torch.save(checkpoint, path)

def main():
    args = arg_parse()
    train_transform = get_transforms(train=True)
    valid_transform = get_transforms(train=False)

    trainloader = get_loader(args.data_dir + '/train', train_transform)
    validloader = get_loader(args.data_dir + '/valid', valid_transform, shuffle=False)

    device = check_device(args.gpu)
    model = load_model(args.arch)
    model.fc = initialize_classifier(model, args.hidden_units, args.output_features)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)

    trained_model = train_model(model, trainloader, validloader, device, optimizer, criterion, args.epochs, print_every=10)
    save_checkpoint(trained_model, args.save_dir, args.arch, args.hidden_units, args.output_features, trainloader.dataset.class_to_idx)

if __name__ == '__main__':
    main()