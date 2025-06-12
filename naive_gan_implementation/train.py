# train.py (outline)
import argparse, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import PhotoDataset
from models import Generator, Discriminator
from torch.utils.tensorboard import SummaryWriter

import os


CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   type=str, required=True)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs',     type=int, default=1)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--split',      type=str, choices=['train','val','test'], default='train')
    p.add_argument('--λ_l1',       type=float, default=100.0)
    p.add_argument('--λ_aes',      type=float, default=1.0)
    p.add_argument('--λ_adv',      type=float, default=1.0)
    return p.parse_args()

def train():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    writer = SummaryWriter("runs/dped_experiment")
    global_step = 0

    ds = PhotoDataset(args.data_dir, split=args.split, size=128)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2)

    G = Generator().to(device)
    D = Discriminator().to(device)
    for p in D.backbone.parameters(): p.requires_grad = False

    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5,0.999))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5,0.999))

    crit_L1  = nn.L1Loss()
    crit_BCE = nn.BCEWithLogitsLoss()
    crit_MSE = nn.MSELoss()

    for epoch in range(args.epochs):
        for raw, edit in loader:
            raw, edit = raw.to(device), edit.to(device)

            # --- train D ---
            fake = G(raw).detach()
            real_logits = D(edit)
            fake_logits = D(fake)
            loss_D = crit_BCE(real_logits, torch.ones_like(real_logits)) \
                   + crit_BCE(fake_logits, torch.zeros_like(fake_logits))
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            writer.add_scalar("Loss/D", loss_D.item(), global_step)
            global_step += 1

            # --- train G ---
            fake = G(raw)
            adv_loss = crit_BCE(D(fake), torch.ones_like(fake_logits))
            l1_loss  = crit_L1(fake, edit)
            aes_loss = crit_MSE(D(fake).mean(1), D(edit).mean(1))
            loss_G = args.λ_adv*adv_loss + args.λ_l1*l1_loss + args.λ_aes*aes_loss
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
            writer.add_scalar("Loss/G", loss_G.item(), global_step)
            global_step += 1

        print(f"Epoch {epoch}  Loss_D {loss_D.item():.3f}  Loss_G {loss_G.item():.3f}")
        torch.save(G.state_dict(), f"{CHECKPOINT_DIR}/G_epoch{epoch}.pth")

    writer.close()


if __name__=='__main__':
    train()
    
