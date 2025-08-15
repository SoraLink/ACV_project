import argparse

from thop import profile, clever_format
import torch.nn
import torchvision.datasets
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTConfig
import torchvision.transforms as T
import time
import matplotlib.pyplot as plt

from huggingface_vit import ViT

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),        # 随机裁剪并填充
    T.RandomHorizontalFlip(),           # 水平翻转
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),  # CIFAR-10 统计均值和方差
])

#zitao_test2
# 测试集数据处理
test_transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010)),
])

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--intermediate_size', type=int, default=256*4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--max_norm', type=int, default=1)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    return parser.parse_args()

def evaluate(model, data_loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.

    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.cuda(), labels.cuda()
            out = model(images)
            loss = loss_fn(out, labels)
            loss_sum += loss.item() * images.size(0)

            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = loss_sum / total
    acc = correct / total
    return acc, avg_loss

def main():
    args = parse()

    cfg = ViTConfig(
        image_size=32,          # CIFAR-10
        patch_size=4,           # 32/4=8 → 序列长度 8*8+1=65 # [2,4,8]
        num_channels=3,
        hidden_size=256,        # d_model [96,192,256]
        num_hidden_layers=8,    #[4,6,12]
        num_attention_heads=4,  # 256/4=64 → head_dim[1,3,8,96]
        intermediate_size=256*4,# MLP比例=4
        qkv_bias=True,
        hidden_act="gelu",
        dropout=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        num_labels=10
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=train_transform, download=True
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=test_transform, download=True
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ViT(cfg)
    model.cuda()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.eval().cuda()
    dummy = torch.randn(1, cfg.num_channels, cfg.image_size, cfg.image_size, device='cuda')
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy,), verbose=False)  # MACs ≈ FLOPs/2
    fwd_flops = macs * 2
    TRAIN_FACTOR = 3.0
    total_train_flops = fwd_flops * TRAIN_FACTOR * len(train_dataset) * args.epochs
    fwd_str, total_str, params_str = clever_format([fwd_flops, total_train_flops, params], "%.3f")
    print(f"[FLOPs] per-image forward={fwd_str}, total training≈{total_str}, params={params_str}")
    model.train()

    best_acc = 0.0
    start_time = time.time()
    print('Start training')

    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", dynamic_ncols=True)

        for images, labels in progress:
            images = images.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            if args.max_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()
            # 统计本epoch的 train loss/acc
            bsz = images.size(0)
            running_loss += loss.item() * bsz
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += bsz
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        # 本epoch的训练指标
        train_loss_epoch = running_loss / max(1, total)
        train_acc_epoch = correct / max(1, total)

        # 验证
        val_acc, val_loss = evaluate(model, test_loader)

        # 记录曲线
        hist['train_loss'].append(train_loss_epoch)
        hist['train_acc'].append(train_acc_epoch)
        hist['val_loss'].append(val_loss)
        hist['val_acc'].append(val_acc)

        print(f"[Epoch {epoch:03d}/{args.epochs}] "
              f"train_loss={running_loss:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "config": model.config.to_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
            }, args.checkpoint_path)

    print(f"Training finished. Best val acc = {best_acc * 100:.2f}%")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training finished. Best val acc = {best_acc * 100:.2f}%")
    print(f"Total training time: {total_time:.2f} s")
    print(f"[FLOPs] total training≈{clever_format([total_train_flops], '%.3f')[0]}")

    epochs = list(range(1, args.epochs + 1))

    # Loss 曲线
    plt.figure()
    plt.plot(epochs, hist['train_loss'], label='train')
    plt.plot(epochs, hist['val_loss'], label='val')
    plt.xlabel('Epoch');
    plt.ylabel('Loss');
    plt.title('Loss Curve');
    plt.legend();
    plt.tight_layout()
    plt.savefig('training_curves_loss.png', dpi=150)
    plt.close()

    # Accuracy 曲线
    plt.figure()
    plt.plot(epochs, hist['train_acc'], label='train')
    plt.plot(epochs, hist['val_acc'], label='val')
    plt.xlabel('Epoch');
    plt.ylabel('Accuracy');
    plt.title('Accuracy Curve');
    plt.legend();
    plt.tight_layout()
    plt.savefig('training_curves_acc.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    main()