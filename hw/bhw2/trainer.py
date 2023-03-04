import os
from tqdm import tqdm
import logging
from datetime import datetime

import torch
import torch.nn as nn

from data import TextDataset
from model import get_model, create_mask


def get_optimizer(model, config):
    optimizer_config = config["optimizer"]
    return torch.optim.Adam(model.parameters(), lr=float(optimizer_config["lr"]), betas=(optimizer_config["beta1"], optimizer_config["beta2"]), eps=float(optimizer_config["eps"]))

def get_datetime():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


class Trainer:
    def __init__(self, config, vocab_sizes, max_length, device, run_name):
        self.model = get_model(config, vocab_sizes, max_length)
        self.model = nn.DataParallel(self.model, config["device_ids"]).to(device)

        self.optimizer = get_optimizer(self.model, config)
        self.criterion = nn.CrossEntropyLoss(ignore_index=TextDataset.PAD_IDX)
        self.device = device

        self.cur_epoch = 0
        self.epochs = config["epochs"]
        self.checkpoint_dir = config["checkpoint"]["dir"]

        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.run_name = run_name
        run_dir = os.path.join(self.checkpoint_dir, run_name)

        if not os.path.exists(run_dir):
            os.mkdir(run_dir)

    def train_epoch(self, loader, tqdm_desc):
        self.model.train()
        losses = 0
        
        for src, dst in tqdm(loader, desc=tqdm_desc):
            self.optimizer.zero_grad()

            src = src.to(self.device)
            dst = dst.to(self.device)

            dst_input = dst[:, :-1]
            
            src_mask, dst_mask, src_padding_mask, dst_padding_mask = create_mask(src, dst_input, self.device)
            logits = self.model(src, dst_input, src_mask, dst_mask, src_padding_mask, dst_padding_mask, src_padding_mask)

            dst_out = dst[:, 1:]

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), dst_out.reshape(-1))

            loss.backward()
            self.optimizer.step()

            losses += loss.item() * src.shape[0]

        return losses / len(loader.dataset)

    @torch.no_grad()
    def validate_epoch(self, loader, tqdm_desc):
        self.model.eval()
        losses = 0

        for src, dst in tqdm(loader, desc=tqdm_desc):
            src = src.to(self.device)
            dst = dst.to(self.device)

            dst_input = dst[:, :-1]
            
            src_mask, dst_mask, src_padding_mask, dst_padding_mask = create_mask(src, dst_input, self.device)
            logits = self.model(src, dst_input, src_mask, dst_mask, src_padding_mask, dst_padding_mask, src_padding_mask)

            dst_out = dst[:, 1:]

            loss = self.criterion(logits.reshape(-1, logits.shape[-1]), dst_out.reshape(-1))

            loss.backward()
            self.optimizer.step()

            losses += loss.item() * src.shape[0]

        return losses / len(loader.dataset)

    def load_from_checkpoint(self, epoch):
        path = self.get_checkpoint_name(epoch) 
        checkpoint = torch.load(path)
        logging.info("Checkpoint is loaded from {} with val_loss {:.5f}".format(path, checkpoint["val_loss"]))

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.cur_epoch = checkpoint["epoch"]
    
    def get_checkpoint_name(self, epoch):
        return os.path.join(self.checkpoint_dir, self.run_name, "epoch_{}.ckpt".format(epoch))

    def save_checkpoint(self, epoch, val_loss):
        path = self.get_checkpoint_name(epoch)

        logging.info("Checkpoint is saved at {}".format(path))

        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            }, path
        )

    def train(self, config, train_loader, val_loader):
        
        for epoch in range(self.cur_epoch, self.epochs):
            train_loss = self.train_epoch(train_loader, f'Training epoch {self.cur_epoch + 1}/{self.epochs}')
            val_loss = self.train_epoch(val_loader, f'Validating epoch {self.cur_epoch + 1}/{self.epochs}')

            logging.info("epoch: {} train_loss: {:.5f} val_loss: {:.5f}".format(epoch + 1, train_loss, val_loss))

            if (epoch + 1) % config["checkpoint"]["step"] == 0:
                self.save_checkpoint(epoch + 1, val_loss)