import os
import torch
import argparse
import torch.nn as nn

from glob import glob
from tqdm import tqdm
from dataset import PetDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from model import UNet, MulticlassDiceLoss
from utils import save_graphs, save_sample_output

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch UNet Training')
    parser.add_argument('--epochs', default=25, type=int, help="Number of training epochs e.g 25")
    parser.add_argument('--batch_size', default=32, type=int, help="Number of images per batch e.g. 256")
    parser.add_argument('--max_pool', action=argparse.BooleanOptionalAction)
    parser.add_argument('--transpose_conv', action=argparse.BooleanOptionalAction)
    parser.add_argument('--cross_entropy_loss', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    return args

def get_unet_type(args):
    unet_type = ""
    if args.max_pool:
        unet_type += "MP"
    else:
        unet_type += "StrConv"
    if args.transpose_conv:
        unet_type += "_Tr"
    else:
        unet_type += "_Ups"
    if args.cross_entropy_loss:
        unet_type += "_CE"
    else:
        unet_type += "_Dice_Loss"
    return unet_type

def _train(model, device, train_loader, optimizer, criterion, train_losses):
    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        pred = model(data)
        
        loss = criterion(pred, target)

        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx}')
    
    train_losses.append(train_loss/len(train_loader))

def _test(model, device, test_loader, criterion, test_losses):
    # Set model to eval
    model.eval()
    test_loss = 0

    # Disable gradient calculation
    with torch.no_grad():
        # iterate though data and calculate loss
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) # Store data and target to device
            pred = model(data)
        
            loss = criterion(pred, target)
            test_loss += loss.item()

    test_loss /= len(test_loader)
    test_losses.append(test_loss)

    # Print results
    print('Test set: Average loss: {:.4f}'.format(test_loss))

def start_training(num_epochs, model, device, train_loader, test_loader, optimizer, criterion):
    train_losses = []
    test_losses = []
    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        _train(model, device, train_loader, optimizer, criterion, train_losses)
        _test(model, device, test_loader, criterion, test_losses)
    return train_losses, test_losses

def main():
    args = get_args()

    os.makedirs('images', exist_ok=True)

    unet_type = get_unet_type(args)

    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    
    image_transform = transforms.Compose([transforms.Resize((240, 240)), 
                                    transforms.ToTensor()])
    
    mask_transform = transforms.Compose([transforms.PILToTensor(),
                                         transforms.Resize((240, 240)),
                                         transforms.Lambda(lambda x: (x - 1).squeeze().type(torch.LongTensor))])

    train_dataset = PetDataset(root='./data', split='trainval', image_transform=image_transform, 
                               mask_transform=mask_transform, download=True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = PetDataset(root='./data', split='test', image_transform=image_transform, 
                              mask_transform=mask_transform, download=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if cuda else "cpu")
    model = UNet(32, args.max_pool, args.transpose_conv, 3)
    model =  model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    if args.cross_entropy_loss:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = MulticlassDiceLoss(num_classes=3, softmax_dim=1)

    train_losses, test_losses = start_training(args.epochs, model, device, train_dataloader, test_dataloader,
                                               optimizer, criterion)
    
    save_graphs(train_losses, test_losses, f'images/{unet_type}_metrics.png')
    save_sample_output(model, test_dataloader, device, f'images/{unet_type}_results.png', 10)
    torch.save(model.state_dict(), f'{unet_type}_unet.pth')

if __name__ == "__main__":
    main()