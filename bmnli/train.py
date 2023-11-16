import argparse
import os
import gc
import glob
import numpy as np
import torch
from tensorboard_logger import configure, log_value
from torch.optim import Adam
from sklearn.model_selection import train_test_split

from dataloader import get_static_loader
from dataset import create_dataset
from evaluate import evaluate
from model import get_classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Biomedical NLI')
    # dataset
    parser.add_argument('--train_json', type=str, help='training json file')
    parser.add_argument('--cr_dir', type=str, help='clinical records directory')
    parser.add_argument('--split_size', type=float, default=0.1, help='train-test split size')

    # training
    parser.add_argument('--output_folder_name', type=str, default='debug')
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=40)

    # others
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--seed', type=int, default=0, help='Seed data split')
    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--debug', type=bool, default=False)

    # checkpoints
    parser.add_argument('--checkpoint_freq', type=int, default=None, help='Checkpoint every N steps')
    parser.add_argument('--skip_model_save', action='store_true')
    args = parser.parse_args()

    n_rows = None
    if args.debug:
        n_rows = 10
        args.batch_size = 2
        args.input_len = 8
        args.num_epochs = 1

    # create model
    model, tokenizer = get_classifier(args.model_name, num_labels=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)

    # create dataset
    cr_json_files = glob.glob(args.cr_dir + "/*.json")
    df = create_dataset(args.train_json, cr_json_files)
    df = df.iloc[:n_rows]

    df_train, df_dev = train_test_split(df, test_size=args.split_size, random_state=args.seed)

    train_loader = get_static_loader(
        tokenizer, df_train['Inp'].values, df_train['Label'].values[:, None],
        max_length=args.input_len, batch_size=args.batch_size, shuffle=True)
    val_loader = get_static_loader(
        tokenizer, df_dev['Inp'].values, df_dev['Label'].values[:, None],
        max_length=args.input_len, batch_size=args.batch_size)
    del df, df_train

    best_val, global_step = 0, 0
    steps_per_epoch = len(train_loader)

    def train_step(batch, device, optimizer, model):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        labels = batch['labels']
        del batch['labels']

        outputs = model(**batch)
        logits = outputs.logits
        loss_ = torch.nn.BCELoss()(torch.nn.Sigmoid()(logits), labels)
        loss_.backward()
        optimizer.step()
        return loss_.detach().cpu().numpy()

    configure(os.path.join(args.output_dir, 'run'))  # tensorboard config
    for epoch in range(args.num_epochs):
        model.train()
        train_loss = []
        gc.collect()
        for batch in train_loader:
            loss = train_step(batch, device, optimizer, model)
            train_loss.append(loss)
            global_step += 1

        gc.collect()
        val, thresh = evaluate(model, val_loader, df_dev['Label'].values)

        if val > best_val:
            print(f"{epoch} New best val score: {val} over {best_val}")
            best_val = val
            best_ep = epoch
            best_thresh = thresh
            state = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'score': val,
                'thresh': thresh
            }
            torch.save(state, os.path.join(args.output_dir, 'model.pt'))

        # tensorboard logger
        log_value('val_f1', val, global_step)
        log_value('train_loss', np.mean(train_loss), global_step)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
