from model import *
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
import copy

def train(train_dataset, val_dataset, args):
    model = PREDAC_CNN(train_dataset.x.shape[2]).to(args.device)
    best_model = model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    pb_miniters = 20
    pbar = tqdm(range(1, args.epochs + 1), miniters=pb_miniters)
    best_val_loss = 1e6
    train_loader = DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
    val_x = torch.tensor(val_dataset.x).float().to(args.device)
    val_y = torch.eye(2)[val_dataset.y].to(args.device)
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            labels = torch.eye(2)[labels.long()].to(args.device)
            images = images.float().to(args.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()

            clip_grad_value_(model.parameters(), clip_value=0.5)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        pbar.set_description(f'Epoch {epoch}, Loss: {epoch_loss:.4f}')
            
        model.eval()
        val_outputs = model(val_x)
        val_loss = criterion(val_outputs, val_y).item()
        if val_loss < best_val_loss:
            best_model = copy.deepcopy(model)
            best_val_loss = val_loss
            
    return best_model

def test(test_dataset, model, args):
    model.eval()
    with torch.no_grad():
        y_pred = F.softmax(model(torch.tensor(test_dataset.x).float().to(args.device)), dim=1)
    return y_pred.cpu().detach().numpy()[:,1].squeeze()

def evaluate_func(dataset, train_idx, val_idx, test_idx, args):
    optied_model = train(dataset.subset(train_idx), dataset.subset(val_idx), args)
    return test(dataset.subset(test_idx), optied_model, args)