import sys
import torch
from tqdm import tqdm
import time

sys.path.insert(0, '../')


def train(train_dataloader, valid_dataloader, model, optimizer, scaler,
          criterion, lr_scheduler, logger, cfg):
    best_epoch = None
    try:
        for epoch in range(cfg.train.max_epochs):
            train_loss = train_epoch(train_dataloader, model, criterion,
                                     scaler, optimizer, cfg)
            if logger:
                logger.train_loss.append(train_loss)
            print('train loss', train_loss)
            valid_loss = eval_epoch(valid_dataloader, model, criterion, scaler, cfg, logger)
            print('valid_loss', valid_loss)
            lr_scheduler.step()
            if logger:
                logger.print_stat_readable(epoch)
                best_epoch = logger.save_model(model.state_dict(), epoch)
                if epoch - best_epoch > 2:
                    print('Stopping criterion were met!')
                    break
    except KeyboardInterrupt:
        pass
    return best_epoch, model


def train_epoch(dataloader, model, criterion, scaler, optimizer, cfg):
    train_loss = 0
    model.train()
    t = 0
    for train_data, train_label, i in (pbar := tqdm(dataloader)):
        train_data = train_data.type(torch.float).to(cfg.device)
        train_label = train_label.type(torch.float).to(cfg.device)
        train_data = scaler.transform(train_data)
        train_label = scaler.transform(train_label)

        optimizer.zero_grad()
        output = model(train_data)

        loss = criterion(output, train_label)

        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=50.0)
        optimizer.step()

        l = loss.item()
        train_loss += l
        pbar.set_description(f'{l}')

    return train_loss / len(dataloader)


def eval_epoch(dataloader, model, criterion, scaler, cfg, logger=None):
    with torch.no_grad():
        model.eval()
        valid_loss = 0.0
        for valid_data, valid_label, i in tqdm(dataloader):
            valid_data = valid_data.type(torch.float).to(cfg.device)
            valid_label = valid_label.type(torch.float).to(cfg.device)
            valid_label = scaler.transform(valid_label, 2)
            valid_data = scaler.transform(valid_data, 2)

            output = model(valid_data)

            loss = criterion(output, valid_label)
            valid_loss += loss.item()
            if logger:
                logger.accumulate_stat(loss.item())
        valid_loss = valid_loss / len(dataloader)
    return valid_loss