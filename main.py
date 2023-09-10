import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric import utils
from networks import Net
import torch.nn.functional as F
import argparse
import os
from torch.utils.data import random_split
from utils import results_to_compare, results_to_file

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=777,
                    help='seed')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0001,
                    help='weight decay')
parser.add_argument('--nhid', type=int, default=128,
                    help='hidden size')
parser.add_argument('--pooling_ratio', type=float, default=0.5,
                    help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.5,
                    help='dropout ratio')
parser.add_argument('--dataset', type=str, default='NCI1',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/COLLAB')
parser.add_argument('--epochs', type=int, default=100000,
                    help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=50,
                    help='patience for earlystopping')
parser.add_argument('--pooling_layer_type', type=str, default='GCNConv',
                    help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/COLLAB')
parser.add_argument('--sampling_method', type=str, default='NUCLEUS',
                    help='TOPK/NUCLEUS')

args = parser.parse_args()
args.device = 'cpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:1'
dataset = TUDataset(os.path.join('data', args.dataset), name=args.dataset)
args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

num_training = int(len(dataset)*0.8)
num_val = int(len(dataset)*0.1)
num_test = len(dataset) - (num_training+num_val)
training_set, validation_set, test_set = random_split(
    dataset, [num_training, num_val, num_test])


train_loader = DataLoader(
    training_set, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(
    validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=16,
                         shuffle=False)

model = Net(args).to(args.device)

optimizer = torch.optim.Adam(
    model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train(epoch):
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(args.device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()

        total_loss += loss.item() * data.num_graphs
        optimizer.step()
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0.
    loss = 0.
    for data in loader:
        data = data.to(args.device)
        out = model(data)
        pred = out.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        loss += F.nll_loss(out, data.y, reduction='sum').item()
    return correct / len(loader.dataset)


run_name = f"{args.dataset}"
run_method = f"{args.sampling_method}"
run_ratio = f"{args.pooling_ratio}"
args.save_path = f"exps/{run_name}/{run_method}/ratio-{run_ratio}"
os.makedirs(os.path.join(args.save_path, str(args.seed)), exist_ok=True)

best_val = 0
patience = 0

for epoch in range(1, args.epochs+1):
    loss = train(epoch)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    state_dict = {"model": model.state_dict(
    ), "optimizer": optimizer.state_dict(), "epoch": epoch}

    if best_val < val_acc:
        best_val = val_acc
        torch.save(state_dict, os.path.join(
            args.save_path, str(args.seed), "best_model.pt"))
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}, '
              f'Test: {test_acc:.4f}')
        patience = 0
    else:
        patience += 1
    if patience > args.patience:
        break


state_dict = torch.load(os.path.join(
    args.save_path, str(args.seed), 'best_model.pt'))
model.load_state_dict(state_dict["model"])
test_acc = test(test_loader)
print("Seed{} Test accuarcy:{}".format(args.seed, test_acc))

results_to_file(args, test_acc)

if args.seed == 786:
    results_to_compare(args)
