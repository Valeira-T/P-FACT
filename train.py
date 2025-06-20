import os
import os.path as op
import pandas as pd
import numpy as np
import torch
from torch.cuda.amp import GradScaler
import argparse
import logging

from model import GPT
from utils.lr import LR
from utils.optim import build_optimizer
from utils.seed import set_seed
from dataset import build_dataloader, SmileDataset
from utils.io import load_config
import torch.multiprocessing as mp


logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, step, ckpt_dir, rng_state):
    os.makedirs(ckpt_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, 'module') else model
    ckpt = {
        'model_state': raw_model.state_dict(),
        'opt_state': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'rng_state': rng_state,
        'cuda_rng': torch.cuda.get_rng_state_all()
    }
    path = op.join(ckpt_dir, f'ckpt_epoch{epoch:03d}_step{step:06d}.pt')
    torch.save(ckpt, path)
    logger.info(f"[Checkpoint] Saved to {path}")


def load_latest_checkpoint(model, optimizer, ckpt_dir, device):
    if not op.isdir(ckpt_dir):
        return 0, 0
    files = [f for f in os.listdir(ckpt_dir) if f.startswith('ckpt_') and f.endswith('.pt')]
    if not files:
        return 0, 0
    files.sort()
    latest = files[-1]
    path = op.join(ckpt_dir, latest)
    logger.info(f"[Resume] Loading checkpoint {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['opt_state'])
    # torch.set_rng_state(ckpt['rng_state'])
    # torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    return ckpt['epoch'], ckpt['step']


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = config.DEVICE
        self.config = config.TRAIN
        self.lr = LR(self.config.LR.LEARNING_RATE)
        self.train_loader = build_dataloader(train_dataset, True, self.config.BATCH_SIZE, self.config.NUM_WORKERS)
        self.valid_loader = build_dataloader(test_dataset, False, self.config.BATCH_SIZE, self.config.NUM_WORKERS)
        self.model = model.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config)
        self.scaler = GradScaler()
        self.best_loss = float('inf')
        self.conditional = True if config.MODEL.NUM_PROPS > 0 else False
        set_seed(config.RANDOM_SEED)

        # resume logic
        ckpt_dir = op.dirname(self.config.CKPT_PATH)
        start_epoch, start_step = load_latest_checkpoint(self.model, self.optimizer, ckpt_dir, self.device)
        self.start_epoch = start_epoch
        self.global_step = start_step
        # tokens processed so far
        self.trained_tokens = start_step

    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        for it, data in enumerate(self.train_loader):
            if self.conditional:
                x, y, p = data
                x, y, p = x.to(self.device), y.to(self.device), p.to(self.device)
            else:
                x, y = data
                x, y, p = x.to(self.device), y.to(self.device), None

            with torch.cuda.amp.autocast():
                logits, loss = self.model(x, y, p)
                loss = loss.mean()
                losses.append(loss.item())

            self.model.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.OPTIM.GRAD_NORM_CLIP)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update lr scheduler
            self.trained_tokens += (y >= 0).sum().item()
            self.lr.step(self.trained_tokens)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.lr.lr

            self.global_step += 1
            if it % 100 == 0:
                print(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {self.lr.lr:e}")

            # periodic checkpoint
            freq = getattr(self.config, 'CKPT_FREQ', 10000)
            if self.global_step % freq == 0:
                save_checkpoint(self.model, self.optimizer,
                                epoch, self.global_step,
                                op.dirname(self.config.CKPT_PATH),
                                torch.get_rng_state())
        return float(np.mean(losses))

    def valid_epoch(self, epoch):
        self.model.eval()
        losses = []
        for data in self.valid_loader:
            if self.conditional:
                x, y, p = data
                x, y, p = x.to(self.device), y.to(self.device), p.to(self.device)
            else:
                x, y, p = data[0].to(self.device), data[1].to(self.device), None
            with torch.cuda.amp.autocast(), torch.no_grad():
                _, loss = self.model(x, y, p)
                losses.append(loss.mean().item())
        test_loss = float(np.mean(losses))
        logger.info("validation loss: %f", test_loss)
        return test_loss

    def train(self):
        for epoch in range(self.start_epoch, self.config.MAX_EPOCHS):
            train_loss = self.train_epoch(epoch)
            valid_loss = self.valid_epoch(epoch)
            print(f"epoch {epoch+1} valid loss {valid_loss:.5f}")
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                print(f"Saving best model at epoch {epoch+1}")
                save_checkpoint(self.model, self.optimizer,
                                epoch, self.global_step,
                                op.dirname(self.config.CKPT_PATH),
                                torch.get_rng_state())

if __name__ == '__main__':
    # 使用 spawn 启动 DataLoader 子进程
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.yaml', help='specify config file')
    args = parser.parse_args()

    config = load_config(args.config)
    train_df = pd.read_csv(config.TRAIN.TRAIN_SET)
    valid_df = pd.read_csv(config.TRAIN.VALID_SET)

    train_dataset = SmileDataset(
        config, dataframe=train_df,
        block_size=config.MODEL.MAX_LEN,
        num_props=config.MODEL.NUM_PROPS,
        aug_prob=config.TRAIN.AUG_PROB
    )
    valid_dataset = SmileDataset(
        config, dataframe=valid_df,
        block_size=config.MODEL.MAX_LEN,
        num_props=config.MODEL.NUM_PROPS,
        aug_prob=0.0
    )

    model = GPT(config)
    trainer = Trainer(model, train_dataset, valid_dataset, config)
    trainer.train()


# **主要改动**:
# 1. **添加 `save_checkpoint` 与 `load_latest_checkpoint`**: 支持脚本重启后从最新检查点恢复。
# 2. **训练循环中周期性保存**: 按 `config.TRAIN.CKPT_FREQ` 步保存检查点。
# 3. **初始化加载**: 在 `Trainer.__init__` 中自动加载已有检查点，恢复 `start_epoch` 与 `global_step`。
