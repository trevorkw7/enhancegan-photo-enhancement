import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as tvmodels              # for VGG
from datasets import PhotoDataset
from models   import Generator, Discriminator
from torch.optim import RMSprop

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   type=str,   required=True)
    p.add_argument('--batch_size', type=int,   default=8)
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--split',      type=str,   choices=['train','val','test'],
                   default='train')
    p.add_argument('--λ_l1',    type=float, default=100.0)
    p.add_argument('--λ_aes',   type=float, default=1.0)
    p.add_argument('--λ_adv',   type=float, default=1.0)
    p.add_argument('--λ_feat',  type=float, default=0.05)
    p.add_argument('--λ_hinge', type=float, default=0.10)
    p.add_argument('--λ_gp',    type=float, default=10.0)   # WGAN-GP gradient penalty weight
    return p.parse_args()

def train():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Models
    G = Generator().to(device)
    D = Discriminator().to(device)
    # ── Critic backbone is trainable for WGAN - GP ─────────────
    # (uncomment below if you previously froze it)
    # for p in D.backbone.parameters():
    #     p.requires_grad = True

    # VGG‐16 for perceptual loss (frozen)
    vgg = tvmodels.vgg16(weights="IMAGENET1K_V1").features[:31].to(device)
    for p in vgg.parameters():
        p.requires_grad = False
    crit_feat = nn.MSELoss()

    # ── Optimizers: RMSProp as per WGAN training ───────────────
    opt_G = RMSprop(G.parameters(), lr=5e-5)
    opt_D = RMSprop(D.parameters(), lr=5e-7)


    # Loss functions
    crit_L1  = nn.L1Loss()
    crit_BCE = nn.BCEWithLogitsLoss()
    crit_MSE = nn.MSELoss()

    # Data loader
    ds     = PhotoDataset(args.data_dir, split=args.split, size=128)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=True, num_workers=2)

    # TensorBoard
    writer = SummaryWriter("runs/dped_experiment")
    global_step = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        epoch_loss_D = 0.0
        epoch_loss_G = 0.0
        n_batches    = 0

        for raw, edit in loader:
            raw, edit = raw.to(device), edit.to(device)

            # ── Train Discriminator (Wasserstein-GP) ─────────────
            fake         = G(raw).detach()
            real_score   = D(edit).mean()
            fake_score   = D(fake).mean()
            # gradient penalty
            α            = torch.rand(raw.size(0),1,1,1, device=device)
            mix          = α * edit + (1 - α) * fake
            mix.requires_grad_(True)
            grad_m       = torch.autograd.grad(
                                 outputs=D(mix).sum(), inputs=mix,
                                 create_graph=True, retain_graph=True
                             )[0]
            gp           = ((grad_m.norm(2, dim=1) - 1) ** 2).mean()
            loss_D       = fake_score - real_score + args.λ_gp * gp
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            writer.add_scalar("Loss/D", loss_D.item(), global_step)
            global_step += 1

            # ── Train Generator (Wasserstein objective) ─────────
            fake      = G(raw)
            adv_loss  = - D(fake).mean()
            l1_loss   = crit_L1(fake, edit)
            aes_loss  = crit_MSE(D(fake).mean(1), D(edit).mean(1))

            # VGG perceptual loss
            feat_loss = crit_feat(vgg(fake), vgg(edit))

            # Aesthetic hinge loss
            fW_fake = D(fake).mean(1)
            fW_raw  = D(raw).mean(1)
            hinge   = torch.clamp(fW_fake - fW_raw, min=0).pow(2).mean()

            # Combine
            loss_G = (
                args.λ_adv   * adv_loss +
                args.λ_l1    * l1_loss +
                args.λ_aes   * aes_loss +
                args.λ_feat  * feat_loss +
                args.λ_hinge * hinge
            )
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            writer.add_scalar("Loss/G_total", loss_G.item(), global_step)
            writer.add_scalar("Loss/feat",     feat_loss.item(), global_step)
            writer.add_scalar("Loss/hinge",    hinge.item(),     global_step)
            global_step += 1

            epoch_loss_D += loss_D.item()
            epoch_loss_G += loss_G.item()
            n_batches    += 1

        # epoch end
        avg_D = epoch_loss_D / n_batches
        avg_G = epoch_loss_G / n_batches
        print(f"Epoch {epoch}  Loss_D {avg_D:.3f}  Loss_G {avg_G:.3f}")

        torch.save(G.state_dict(),
                   f"{CHECKPOINT_DIR}/G_epoch{epoch}.pth")
        torch.save(D.state_dict(),
                   f"{CHECKPOINT_DIR}/D_epoch{epoch}.pth")

    writer.close()
    print("Training complete!")

if __name__ == '__main__':
    train()