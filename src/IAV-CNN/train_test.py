from model import *
import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

def train(train_dataset, val_dataset, args):
    model = IAV_CNN().to(args.device)
    best_model = model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    pb_miniters = 20
    pbar = tqdm.tqdm(range(1, args.epochs + 1), miniters=pb_miniters)
    best_val_loss = 1e6
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in pbar:
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            labels = torch.eye(2)[labels.long()].to(args.device)
            images = images.float().to(args.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            all_preds = []
            all_labels = []
            for i, (images, labels) in enumerate(val_loader):
                images = images.float().to(args.device)
                outputs = model(images)
                all_preds.append(outputs)
                all_labels.append(labels)
            flat_preds = torch.cat(all_preds, dim=0) 
            flat_labels = torch.cat(all_labels, dim=0)
            flat_labels_onehot = torch.eye(2)[flat_labels.long()].to(args.device)
            val_loss = criterion(flat_preds, flat_labels_onehot).item()
            if val_loss < best_val_loss:
                best_model = copy.deepcopy(model)
                best_val_loss = val_loss
            
    return best_model


def test(test_dataset, model, args):
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    with torch.no_grad():
        all_preds = []
        for i, (images, _) in enumerate(test_loader):
            images = images.float().to(args.device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1) 
            all_preds.append(probs)
        y_pred = torch.cat(all_preds, dim=0)
    return y_pred.cpu().detach().numpy()[:,1].squeeze()

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    optied_model = train(dataset.subset(train_idx), dataset.subset(val_idx), args)
    return test(dataset.subset(test_idx), optied_model, args)