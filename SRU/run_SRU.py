import torch
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import cuda
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

class SRU(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        super(SRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sigmoid = nn.Sigmoid()
        self.linear = nn.Linear(hidden_size, output_size)
        self.init_hidden()
    
    def init_hidden(self):
        
        self.W = nn.Parameter(torch.Tensor(self.input_size, 3 * self.hidden_size))
        self.bias = nn.Parameter(torch.Tensor(2 * self.hidden_size))
        self.v = nn.Parameter(torch.Tensor(2 * self.hidden_size))
        
        # initialization based on https://arxiv.org/abs/1709.02755
        val_range = (3.0 / self.input_size) ** 0.5
        self.W.data.uniform_(-val_range, val_range)
        self.bias.data.zero_()
        self.v.data.zero_()
    
    def forward(self, input, c0=None):
        
        seq_len, batch_size, input_size = input.shape
        assert input_size == self.input_size, 'input_size = {}, self.input_size = {}'.format(input_size, self.input_size)
        
        if c0 is None:
           c0 = torch.zeros(batch_size, input_size)

        # shape (seq_len, batch_size, 3 * hidden_size)
        U = input @ self.W
        c = c0
        H = []
        C = []
        for l in range(seq_len):
            f = self.sigmoid(U[l, :, self.hidden_size: 2 * self.hidden_size] \
                + self.v[:self.hidden_size] * c + self.bias[:self.hidden_size])
            c = f * c + (1 - f) * U[l, :, :self.hidden_size]
            r = self.sigmoid(U[l, :, 2 * self.hidden_size:] + \
                self.v[self.hidden_size:] * c + self.bias[self.hidden_size:])
            h = r * c + (1 - r) * input[l, :, :]
            C.append(c)
            H.append(h)
        C = torch.stack(C)
        H = torch.stack(H)
        return self.linear(H)

class CustomDataset(Dataset):

    def __init__(self, data, seq_len):

        self.data = data
        self.max_len = data.size(0)
        self.seq_len = seq_len

    def __len__(self):
        return self.max_len - self.seq_len

    def __getitem__(self, index):
        
        input = self.data[index: index + self.seq_len]
        target = self.data[index + 1: index + self.seq_len + 1]
        
        return input, target

def train(model, device, loader, optimizer, criterion):
    
    print('[Training...]')
    model.train()
    
    losses = []
    
    for _, (input, target) in enumerate(loader):
        
        input = input.permute(1, 0, 2).to(device)
        target = target.permute(1, 0, 2).to(device)
        output = model(input)
        loss = criterion(target, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        print(f'Completed {_} Batches, Loss: {loss.item()}')
        
    return losses

def validate(model, device, loader, criterion):
    
    print('[Validating...]')
    model.eval()
    
    ts_output = []
    ts_pred = []
    losses = []
    
    with torch.no_grad():
        for _, (input, target) in enumerate(loader):

            input = input.permute(1, 0, 2).to(device)
            target = target.permute(1, 0, 2).to(device)
            preds = model(input)
            loss = criterion(target, preds)
            
            ts_output.extend(target[-1, :, -1])
            ts_pred.extend(preds[-1, :, -1])
            losses.append(loss.item())
            
            print(f'Completed {_} Batches')
            
    return ts_output, ts_pred, losses

def plot_result(path, ts_output, ts_pred, seq_len, max_len):
    
    plt.figure(figsize=(10, 5))
    plt.title("full input sequence vs. prediction (dev)")
    plt.plot(ts_output[:max_len], 'b', label='truth')
    plt.plot(list(range(seq_len, max_len)), ts_pred[seq_len: max_len], 'r', label='pred')
    for j in range(seq_len, max_len):
        plt.axvline(x=j)
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_loss(path, loss_train, loss_val):

    plt.title("full loss vs. batches")
    plt.plot(loss_train, 'b', label='training')
    plt.plot(loss_val, 'r', label='val')
    plt.legend()
    plt.savefig(path)
    plt.close()

def main(args):

    # check validity
    if args.max_len < args.seq_len:
        print('max_len = {}, seq_len = {}, need max_len >= seq_len', args.max_len, args.seq_len)
        exit(1)

    # set random seed and device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if cuda.is_available() else 'cpu'
    print('device', device)
    
    # create raw dataset
    data = torch.arange(args.max_len) + 1.
    split = args.max_len - int(args.train_split * args.max_len)
    train_dataset = data[:split].unsqueeze(-1)
    val_dataset = data.unsqueeze(-1)
    
    # create batched dataset
    training_set = CustomDataset(train_dataset, args.seq_len)
    val_set = CustomDataset(val_dataset, args.seq_len)

    # define dataloader parameters
    train_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': args.batch_size,
        'shuffle': False,
        'num_workers': 0
        }

    # create dataloader
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)
    
    # create model
    model = SRU(args.input_size, args.hidden_size, args.output_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    criterion = nn.L1Loss().to(device)
    
    losses_train = []
    losses_val = []
    for epoch in range(args.epochs):
        print('Epoch', epoch)
        loss_train = train(model, device, training_loader, optimizer, criterion)
        losses_train.extend(loss_train)
        ts_output, ts_pred, loss_val = validate(model, device, val_loader, criterion)
        losses_val.extend(loss_val)
        if (epoch + 1) % 10 == 0:
            plot_result('predictions_{}.png'.format(epoch + 1), ts_output, ts_pred, args.seq_len, args.plot_len)
            final_df = pd.DataFrame({'Actual Values': [item.numpy() for item in ts_output[:args.plot_len]], 
                                    'Predicted Values': [item.numpy() for item in ts_pred[:args.plot_len]]})
            final_df.to_csv('predictions_' + str(epoch + 1) + '.csv', index=False)
    plot_loss('loss.png', losses_train, losses_val)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SRU for sequence prediction", add_help=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=3, help="input sequence length to SRU")
    parser.add_argument("--max_len", type=int, default=128, help="the full input sequence length")
    parser.add_argument("--plot_len", type=int, default=10, help="sequence length to plot for visualization")
    parser.add_argument("--input_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train_split", type=int, default=0.5)
    args = parser.parse_args()
    print(args)
    main(args)