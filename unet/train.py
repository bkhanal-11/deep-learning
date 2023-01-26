# Libraries
import torch
import torchvision

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from tqdm.auto import tqdm
from timeit import default_timer as timer
import warnings

import os
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

class DiceLoss(nn.Module):
    def __init__(self, aerial = False, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.aerial = aerial

    def forward(self, logits, targets):
        if self.aerial:
            if logits.dim()>2:
                logits = logits.view(logits.size(0),logits.size(1),-1)  # N,C,H,W => N,C,H*W
                logits = logits.transpose(1,2)    # N,C,H*W => N,H*W,C
                logits = logits.contiguous().view(-1,logits.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1,1)
        smooth = 1
        num = targets.size(0)
        probs = logits
        m1 = probs.reshape(num, -1)
        m2 = targets.reshape(num, -1)
        intersection = (m1 * m2)

        score = (2. * intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - (score.sum() / num)
        return score

def train_model(model, optimizer, train_dataloader):
    model.train()
    losses = 0
    
    print("Training Phase ...")
    pbar = tqdm(range(len(train_dataloader)))
    for data, labels in train_dataloader:
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(data)

        optimizer.zero_grad()

        loss = loss_fn(logits, labels)
        #loss = nn.CrossEntropyLoss(outputs, y) - torch.log(DiceLoss(outputs, y))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        pbar.update()

    return losses / len(train_dataloader)

def evaluate_model(model, val_dataloader):
    model.eval()
    losses = 0

    print("Validation Phase ...")
    qbar = tqdm(range(len(val_dataloader)))
    for data, labels in val_dataloader:
        data = data.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(data)

        loss = loss_fn(logits, labels)
        #loss = nn.CrossEntropyLoss(outputs, y) - torch.log(DiceLoss(outputs, y))
        losses += loss.item()
        qbar.update()

    return losses / len(val_dataloader)

if __name__ == "__main__":
    parser = ArgumentParser("Train Model using UNet architecture for semantic segmentation.")
    parser.add_argument('--dataset', 
                help='Name of the dataset [0 for Self driving, 1 for Aerial Imaging]')
    args = parser.parse_args()
    
    dataset = str(args.dataset)
    
    if dataset == "0":
        train_dataloader, validation_dataloader, _ = SelfDrivingDataset(batch_size = 8)
        num_filter = 64
        num_classes = 13
        model = UNet(num_filter, num_classes).to(DEVICE)
        loss_fn = DiceLoss()
        # loss_fn = nn.CrossEntropyLoss()
        save_model_path = 'models/UNet_model_self.pth'
    elif dataset == "1":
        train_dataloader, validation_dataloader, _ = AerialImagingDataset(batch_size = 4)
        num_filter = 32
        num_classes = 6
        model = UNet(num_filter, num_classes).to(DEVICE)
        loss_fn = DiceLoss(aerial=True)
        # loss_fn = nn.CrossEntropyLoss()
        save_model_path = 'models/UNet_model_aerial.pth'

    summary(model, input_size=(3, 256, 256))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    warnings.filterwarnings('ignore')
    epochs = 10

    history = []
    for epoch in range(1, epochs + 1):
        start_time = timer()
        train_loss = train_model(model, optimizer, train_dataloader)
        end_time = timer()
        
        val_loss = evaluate_model(model, validation_dataloader)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
        
        history.append({"Epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "epoch_time": (end_time - start_time)})
    
    if not os.path.isdir("models"):
        os.mkdir("models")

    torch.save(model.state_dict(), save_model_path)

    writer = SummaryWriter()

    images, labels = next(iter(train_dataloader))

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)
    writer.add_graph(model, images.to(DEVICE))
    writer.close()
    