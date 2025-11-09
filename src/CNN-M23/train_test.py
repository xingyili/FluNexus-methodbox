from model import *
import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import copy

def train(train_dataset, val_dataset, args):
    model = CNN_M23().to(args.device)
    best_model = model
    criterion = nn.L1Loss()
    lr = 1e-3
    base_decay = 2e-4
    optimizer = optim.SGD([
            {'params': model.conv0.weight, 'lr': lr * 1, 'weight_decay': base_decay * 1},
            {'params': model.conv0.bias, 'lr': lr * 2, 'weight_decay': 0},
            {'params': model.conv1.weight, 'lr': lr * 1, 'weight_decay': base_decay * 1},
            {'params': model.conv1.bias, 'lr': lr * 2, 'weight_decay': 0},
            {'params': model.conv2.weight, 'lr': lr * 1, 'weight_decay': base_decay * 1},
            {'params': model.conv2.bias, 'lr': lr * 2, 'weight_decay': 0},
            {'params': model.fc6.weight, 'lr': lr * 1, 'weight_decay': base_decay * 1},
            {'params': model.fc6.bias, 'lr': lr * 2, 'weight_decay': 0},
            {'params': model.fc_last.weight, 'lr': lr * 1, 'weight_decay': base_decay * 1},
            {'params': model.fc_last.bias, 'lr': lr * 2, 'weight_decay': 0},
        ], lr=lr, momentum=0.9, weight_decay=base_decay)
    
    def poly_lr_lambda(epoch):
        return (1 - epoch / args.epochs) ** 1.0
    
    scheduler = LambdaLR(optimizer, lr_lambda=poly_lr_lambda)
    pb_miniters = 20
    pbar = tqdm.tqdm(range(1, args.epochs + 1), miniters=pb_miniters)
    best_val_loss = 1e6

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
    val_x = torch.tensor(val_dataset.x).float().to(args.device)
    val_y = torch.tensor(val_dataset.y).float().unsqueeze(1).to(args.device)
    for epoch in pbar:
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            labels = labels.float().unsqueeze(1).to(args.device)
            images = images.float().to(args.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y).item()
            if val_loss < best_val_loss:
                best_model = copy.deepcopy(model)
                best_val_loss = val_loss
    return best_model

def test(test_dataset, model, args):
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(test_dataset.x).float().to(args.device))
    return y_pred.cpu().detach().numpy()

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    optied_model = train(dataset.subset(train_idx), dataset.subset(val_idx), args)
    return test(dataset.subset(test_idx), optied_model, args)