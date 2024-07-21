import torch
import numpy as np
import matplotlib.pyplot as plt

def save_sample_output(model, loader, device, path, image_no=2):
    dataiter = iter(loader)

    with torch.no_grad():
        _, axes = plt.subplots(image_no, 3, figsize=(8, 20))
        for i in range(image_no):
            images, labels = next(dataiter)
            images, labels = images.to(device), labels.to(device)

            # Original Image
            ax = axes[i, 0]
            ax.set_title('Original', fontsize=6)
            ax.imshow(np.transpose(images[0].cpu().numpy(), (1, 2, 0)))

            # Ground Truth
            ax = axes[i, 1]
            ax.set_title('Ground Truth', fontsize=6)
            ax.imshow(labels[0].cpu().numpy())

            # Predicted Mask
            output = model(images).squeeze()
            predicted_masks = torch.argmax(output, 1).cpu().numpy()
            ax = axes[i, 2]
            ax.set_title('Predicted', fontsize=6)
            ax.imshow(predicted_masks[0])

    plt.tight_layout()
    plt.savefig(path)
    print(f"Sample output images are saved at {path}")

def save_graphs(train_losses, test_losses, path):
    _, axs = plt.subplots(1,1,figsize=(10,5))
    axs.plot(train_losses, label ='Train')
    axs.plot(test_losses, label ='Test')
    axs.set_title("Loss")
    plt.savefig(path)
    print(f"Training metrics plot is saved at {path}")