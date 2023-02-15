# Libraries
import torch

import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser

from GPUtil import showUtilization as gpu_usage
from numba import cuda

from models import UNet, UNetM

from datasets import SelfDrivingDataset, AerialImagingDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()  

def get_orig(image):
    image = image.permute(1, 2, 0)
    image = image.numpy()
    image = np.clip(image, 0, 1)
    return image

if __name__ == "__main__":
    parser = ArgumentParser("Testing Model using UNet architecture for semantic segmentation.")
    parser.add_argument('--dataset', 
                help='Name of the dataset [0 for Self driving, 1 for Aerial Imaging]')
    args = parser.parse_args()
    
    dataset = str(args.dataset)
    
    if dataset == "0":
        _, _, test_dataloader = SelfDrivingDataset(batch_size = 8)
        num_filter = 64
        num_classes = 13
        model = UNet(num_filter, num_classes).to(DEVICE)
        save_model_path = 'models/UNet_model_self.pth'
        model.load_state_dict(torch.load(save_model_path))

        class_idx = 10

        for i, data in enumerate(test_dataloader):
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            model.eval()
            outputs = model(images)
            f, axarr = plt.subplots(1,3, figsize=(15, 6))

            for j in range(0, 4):
                axarr[0].imshow(outputs.squeeze().detach().cpu().numpy()[j,class_idx,:,:])
                axarr[0].set_title('Predicted Label')
                
                axarr[1].imshow(labels.detach().cpu().numpy()[j,class_idx, :,:])
                axarr[1].set_title('Ground Truth Labels')

                original = get_orig(images[j].cpu())
                axarr[2].imshow(original)
                axarr[2].set_title('Original Images')
                plt.show()
            if i > 3:
                break
    elif dataset == "1":
        _, _, test_dataloader = AerialImagingDataset(batch_size = 4)
        num_filter = 32
        num_classes = 6
        model = UNet(num_filter, num_classes).to(DEVICE)
        save_model_path = 'models/UNet_model_aerial.pth'
        model.eval()
        for i, (X, y) in enumerate(test_dataloader):
            for j in range(0, 4):
                X = X.to(DEVICE)
                y = y.to(DEVICE)
                result = model(X[j:j+1])
                mask = torch.argmax(result, axis=1).cpu().detach().numpy()[0]
                gt_mask = y[j].cpu()

                plt.figure(figsize=(15,6))

                plt.subplot(1,3,1)
                plt.title('Predicted Label')
                plt.imshow(mask)
                
                plt.subplot(1,3,2)
                plt.title('Ground Truth Label')
                plt.imshow(gt_mask)
                
                plt.subplot(1,3,3)
                im = np.moveaxis(X[j].cpu().detach().numpy(), 0, -1).copy()*255
                im = im.astype(int)
                plt.title('Original Images')
                plt.imshow(im)
                
                plt.show()
            if i > 3:
                break
    