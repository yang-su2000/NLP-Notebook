import torch
import argparse
import numpy as np
import pandas as pd
from torch import cuda
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = dataframe['text']
        self.max_len = max_len
        self.task_prefix = 'Translate English to reverse-English: '

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        
        text = self.text[index]
        source = self.tokenizer.encode(self.task_prefix + text[:self.max_len], 
                                padding='max_length',
                                max_length=self.max_len + len(self.task_prefix),
                                truncation=True,
                                return_tensors="pt")
        target = self.tokenizer.encode(text[:self.max_len][::-1] + self.tokenizer.eos_token,
                                padding='max_length',
                                max_length=self.max_len + len(self.tokenizer.eos_token),
                                truncation=True,
                                return_tensors="pt")
        input_ids = source.squeeze()
        labels = target.squeeze()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }

def train(model, device, loader, tokenizer, optimizer):
    
    print('[Training...]')
    model.train()
    
    for _, data in enumerate(loader):

        input_ids = data['input_ids'].to(device)
        labels = data['labels'].to(device)
        labels[labels == tokenizer.pad_token_id] = -100
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs[0]

        if _ % 100 == 0:
            print(f'Completed {_}, Loss: {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(model, device, loader, tokenizer, max_len):
    
    print('[Validating...]')
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for _, data in enumerate(loader):

            input_ids = data['input_ids'].to(device)
            labels = data['labels'].to(device)
            generated_ids = model.generate(input_ids=input_ids, max_length=max_len + len(tokenizer.eos_token))
            preds = [tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True) for t in labels]
            
            if _ % 100 == 0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
            
            break
            
    return predictions, actuals

def main(args):

    # set random seed and device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = 'cuda' if cuda.is_available() else 'cpu'
    print('device', device)
    
    # create raw dataset
    train_dataset = load_dataset('ag_news', split='train[:' + args.train_split + ']')
    val_dataset = load_dataset('ag_news', split='train[' + args.train_split + ':]')
    
    # create tokenized dataset
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    training_set = CustomDataset(train_dataset, tokenizer, args.max_len)
    val_set = CustomDataset(val_dataset, tokenizer, args.max_len)

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
    model = T5ForConditionalGeneration.from_pretrained("google/byt5-small")
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        print('Epoch', epoch)
        train(model, device, training_loader, tokenizer, optimizer)
        predictions_rev, actuals_rev = validate(model, device, val_loader, tokenizer, args.max_len)
        print('Generating Text')
        predictions = [text[::-1] for text in predictions_rev]
        actuals = [text[::-1] for text in actuals_rev]
        final_df = pd.DataFrame({'Actual Text': actuals, 'Generated Text (Reversed)': predictions,
                                 'Actual Text (Reversed)': actuals_rev, 'Generated Text': predictions_rev})
        final_df.to_csv('predictions_' + str(epoch) + '.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="T5 fine-tuning for reverse-english generation", add_help=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_split", type=str, default='80%')
    args = parser.parse_args()
    print(args)
    main(args)