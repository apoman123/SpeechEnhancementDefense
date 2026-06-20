import time
import argparse
import os

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datasets import Audio, load_from_disk, concatenate_datasets
from tqdm import tqdm
from m5 import M5

def zero_mean_normalization(data):
    mean = torch.mean(data, dim=-1, keepdim=True).expand(data.shape)
    std = torch.std(data, dim=-1, keepdim=True).expand(data.shape)
    data = (data - mean) / std
    return data
    
def test(model, data_loader, verbose=False):
    global layer_norm
    """Measures the accuracy of a model on a data set.""" 
    # Make sure the model is in evaluation mode.
    model.eval()
    correct = 0
#     print('----- Model Evaluation -----')
    # We do not need to maintain intermediate activations while testing.
    with torch.no_grad():
        # Loop over test data.
        for features, target in tqdm(data_loader, total=len(data_loader.batch_sampler), desc="Testing"):
            # Forward pass.
            features = zero_mean_normalization(features)
            output = model(features.to(device))
            # Get the label corresponding to the highest predicted probability.
            pred = output.argmax(dim=1, keepdim=True)
            # Count number of correct predictions.
            correct += pred.cpu().eq(target.view_as(pred)).sum().item()
    # Print test accuracy.
    percent = 100. * correct / len(data_loader.sampler)
    if verbose:
        print(f'Test accuracy: {correct} / {len(data_loader.sampler)} ({percent:.0f}%)')
    return percent


def train(model, criterion, data_loader, test_loader, optimizer, num_epochs, args):
    global layer_norm
    """Simple training loop for a PyTorch model.""" 
    
    # Move model to the device (CPU or GPU).
    model.to(device)
    
    accs = []
    # Exponential moving average of the loss.
    ema_loss = None

#     print('----- Training Loop -----')
    # Loop over epochs.
    for epoch in range(num_epochs):
        tick = time.time()
        model.train()
        # Loop over data.
        for batch_idx, (features, target) in tqdm(enumerate(data_loader), total=len(data_loader.batch_sampler), desc="training"):
            # Forward pass.
            features = zero_mean_normalization(features)
            output = model(features.to(device))
            # loss = criterion(output.to(device), target.to(device))
            loss = F.nll_loss(F.log_softmax(output.to(device), dim=-1), target.to(device))
            loss.backward()
            
            if (batch_idx+1) % 5 == 0:
                # Backward pass.
                optimizer.step()
                optimizer.zero_grad()
            # NOTE: It is important to call .item() on the loss before summing.
            if ema_loss is None:
                ema_loss = loss.item()
            else:
                ema_loss += (loss.item() - ema_loss) * 0.01
            tock = time.time()
        acc = test(model, test_loader, verbose=True)
        accs.append(acc)
        # Print out progress the end of epoch.
        print('Epoch: {} \tLoss: {:.6f} \t Time taken: {:.6f} seconds'.format(epoch+1, ema_loss, tock-tick),)
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f"m5net_{epoch}.pt"))
        print("Model Saved!")
        # if os.path.isfile(f'model_{epoch-1}.ckpt'):
        #     os.remove(f'model_{epoch-1}.ckpt')
    return accs


# using glorot initialization
def init_weights(m):
    if isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight.data)
class ToMono(torch.nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.mean(waveform, dim=0, keepdim=True)

class Normalize(torch.nn.Module):
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return (waveform-waveform.mean()) / waveform.std()

class Pad(torch.nn.Module):
    def __init__(self, value: float, size: int):
        super(Pad, self).__init__()
        self.value = value
        self.size = size
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(waveform, (0, self.size-max(waveform.shape)), "constant", self.value)
        
def collate_fn(batch):
    lengths = [data['audio']['array'].shape[-1] for data in batch]
    max_length = max(lengths)
    num_pads = [max_length - length for length in lengths]
    tensors = torch.stack(
        [
            torch.cat([
            torch.from_numpy(data['audio']['array']).float(), torch.zeros(pads)
            ], dim=0)
            for data, pads in zip(batch, num_pads)
        ]
    )
    # tensors = zero_mean_normalization(tensors)
    labels = torch.tensor([data['label'] for data in batch])
    return tensors.unsqueeze(1), labels
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="", type=str)
    parser.add_argument("--log_path", default="", type=str)
    parser.add_argument("--dataset_path")
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    speech_commands = load_from_disk(args.dataset_path).cast_column("audio", Audio(sampling_rate=16000))
    augmented_training_set = load_from_disk("/data/nas07/PersonalData/apoman123/sc09_augmented_training_set")
    
    training_set = concatenate_datasets([speech_commands['train'], augmented_training_set])
    
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        speech_commands["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        speech_commands["validation"],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    model = M5(
        n_input=1,
        n_output=10
    )
    global layer_norm
    layer_norm = nn.LayerNorm(1, elementwise_affine=False)
    print("Num Parameters:", sum([p.numel() for p in model.parameters()]))

    # audio_transform = torch.nn.Sequential(*[
    #     Normalize() # normalize audio signal to have mean=0 & std=1
    # ])
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4) #L2 regularization is added
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    writer = SummaryWriter(log_dir=args.log_path)
    num_epochs = 100
    accs = train(model, criterion, train_loader, test_loader, optimizer, num_epochs=num_epochs, args=args)
    for idx in range(len(accs)):
        writer.add_scalar("Test Acc", accs[idx], idx)
    
    