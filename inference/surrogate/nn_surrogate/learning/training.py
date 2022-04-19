#!/usr/bin/env python

from copy import deepcopy
import torch
from torch.utils.data import DataLoader

def train_model(model,
                data_loader: DataLoader,
                x_valid: torch.tensor,
                y_valid: torch.tensor,
                criterion = torch.nn.MSELoss(),
                lr=0.1,
                max_epoch=5000,
                info_every=100,
                device=torch.device('cpu')):

    model.to(device)

    x_valid, y_valid = x_valid.to(device), y_valid.to(device)

    with torch.no_grad():
        model_bck = deepcopy(model)

    best_valid_loss = 1e20
    num_worse_valid_losses = 0
    patience = 10
    max_number_of_rounds = 5
    number_of_rounds = 0
    lr_reduction_factor = 0.1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # trace
    train_losses = list()
    valid_losses = list()

    for epoch in range(max_epoch):

        y_pred_valid = model(x_valid)

        with torch.no_grad():
            valid_loss = criterion(y_pred_valid, y_valid)

        if valid_loss.item() < best_valid_loss:
            best_valid_loss = valid_loss.item()
            num_worse_valid_losses = 0
            model_bck.load_state_dict(model.state_dict())
        else:
            num_worse_valid_losses += 1

        if num_worse_valid_losses > patience:
            model.load_state_dict(model_bck.state_dict())
            num_worse_valid_losses = 0
            number_of_rounds += 1
            lr *= lr_reduction_factor
            print(f"reduced lr from {lr/lr_reduction_factor:e} to {lr:e}")
            # set the learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


        if number_of_rounds >= max_number_of_rounds:
            break

        train_loss = 0.0
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            batch_loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            batch_loss.backward()

            if epoch > 0:
                optimizer.step()

            train_loss += batch_loss.cpu().detach().numpy()

        nbatches = len(data_loader)
        train_loss /= nbatches

        if epoch % info_every == 0:
            print(f"epoch {epoch:05d}: training loss {train_loss.item():e}, valid loss {valid_loss.item():e}")

        # save trace
        train_losses.append(train_loss.item())
        valid_losses.append(valid_loss.item())

    return model, train_losses, valid_losses
