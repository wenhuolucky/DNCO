import argparse
import datetime
import time
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import losses as L
import models
import torch.nn.functional as F
from dataset import *
from kornia import augmentation
from query_sample import generate_adv, generate_hee, generate_ue
from robust_test import robust_eval
from torchvision import datasets, transforms
from utils import *
import os

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Data-Free Hard-Label Robustness Stealing")

# model configuration
parser.add_argument(
    "--arch",
    type=str,
    choices=["ResNet18", "ResNet34", "WideResNet", "MobileNet"],
    default="ResNet18",
)
parser.add_argument(
    "--target_arch", type=str, default="ResNet18", choices=["ResNet18", "WideResNet"]
)
parser.add_argument(
    "--target_defense",
    type=str,
    default="AT",
    choices=["AT", "TRADES", "STAT_AWP"],
)
parser.add_argument("--target_dir", type=str, default="./checkpoints/")

# generator configuration
parser.add_argument(
    "--gen_dim_z",
    "-gdz",
    type=int,
    default=256,
    help="Dimension of generator input noise.",
)
parser.add_argument(
    "--gen_distribution",
    "-gd",
    type=str,
    default="normal",
    help="Input noise distribution: normal (default) or uniform.",
)

# dataset configuration
parser.add_argument(
    "--data", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"]
)
parser.add_argument(
    "--data_path", type=str, default="~/datasets/", help="where is the dataset CIFAR-10"
)
parser.add_argument(
    "--test_batch_size",
    type=int,
    default=512,
    metavar="N",
    help="input batch size for testing",
)

# training configuration
parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    metavar="N",
    help="input batch size for training",
)
parser.add_argument(
    "--epochs", type=int, default=300, metavar="N", help="number of epochs to train"
)
parser.add_argument(
    "--lr", type=float, default=0.1, metavar="N", help="learning rate of clone model"
)
parser.add_argument(
    "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
)
parser.add_argument(
    "--weight_decay",
    default=1e-4,
    type=float,
)
parser.add_argument(
    "--N_C", type=int, default=500, metavar="N", help="iterations of clone model"
)
parser.add_argument(
    "--N_G", type=int, default=10, metavar="N", help="iterations of generator"
)
parser.add_argument(
    "--lr_G", type=float, default=0.002, metavar="N", help="learning rate of generator"
)
parser.add_argument(
    "--lr_z", type=float, default=0.01, help="learning rate of latent code"
)
parser.add_argument(
    "--lam", type=float, default=3, help="hyperparameter for balancing two loss terms"
)
parser.add_argument(
    "--label_smooth_factor",
    default=0.2,
    type=float,
    help="0.2 for CIFAR 10, 0.02 for CIFAR100",
)

# HEE configuration
parser.add_argument(
    "--lr_hee", type=float, default=0.03, metavar="N", help="number of epochs to train"
)
parser.add_argument("--steps_hee", default=10, type=int, help="perturb number of steps")
parser.add_argument(
    "--query_mode",
    default="HEE",
    type=str,
    choices=[
        "UE",
        "AE",
        "HEE",
        "AT",
    ],
)
# for AE/UE
parser.add_argument("--epsilon", default=8.0 / 255, type=eval)
parser.add_argument("--num_steps", default=10, type=int)
parser.add_argument("--step_size", default=2.0 / 255, type=eval)

# other configuration
parser.add_argument(
    "--result_dir", default="results", help="directory of model for saving checkpoint"
)
parser.add_argument(
    "--save_freq", "-s", default=50, type=int, metavar="N", help="save frequency"
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
)

args = parser.parse_args()

if args.data == "CIFAR100":
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

target_path = os.path.join(
    args.target_dir,
    args.data,
    args.target_defense,
    args.target_arch,
    "best_robust_checkpoint.tar",
)
exp_time = datetime.datetime.now().strftime("%y%m%d_%H%M")
checkpoint_path = os.path.join(
    args.result_dir,
    args.data,
    args.target_defense + "_" + args.target_arch + "-to-" + args.arch,
    args.query_mode,
    exp_time,
    "checkpoints",
)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

logger = Logger(
    os.path.join(
        args.result_dir,
        args.data,
        args.target_defense + "_" + args.target_arch + "-to-" + args.arch,
        args.query_mode,
        exp_time,
        "output.log",
    )
)

if args.data == "CIFAR10" or args.data == "CIFAR100":
    img_size = 32
    img_shape = (3, 32, 32)
    nc = 3

if args.seed is not None:
    random_seed(args.seed)

# Standard Augmentation
std_aug = augmentation.container.ImageSequential(
    augmentation.RandomCrop(size=[img_shape[-2], img_shape[-1]], padding=4),
    augmentation.RandomHorizontalFlip(),
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_nature_acc = 0
best_robust_acc = 0
curr_query_times = 0

class FileDataStorage:
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.unlabeled_files = []  # 未标注池
        self.labeled_files = []    # 已标注池

        self.data_save_path = os.path.join(
            args.result_dir,
            args.data,
            args.target_defense + "_" + args.target_arch + "-to-" + args.arch,
            args.query_mode,
            exp_time,
            "data_storage"
        )
        self.unlabeled_data_dir = os.path.join(self.data_save_path, "unlabeled_data")
        self.labeled_data_dir = os.path.join(self.data_save_path, "labeled_data")
        os.makedirs(self.unlabeled_data_dir, exist_ok=True)
        os.makedirs(self.labeled_data_dir, exist_ok=True)

    def _clean_old_files(self, file_list, dir_path):
        if len(file_list) > self.max_size:
            def extract_timestamp(path):
                filename = os.path.basename(path[0])
                timestamp_str = filename.split('_')[1]
                return datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
            file_list.sort(key=lambda x: extract_timestamp(x))
            for f_tuple in file_list[:-self.max_size]:
                fake_path, labels_path = f_tuple
                if os.path.exists(fake_path):
                    os.remove(fake_path)
                if os.path.exists(labels_path):
                    os.remove(labels_path)
            file_list = file_list[-self.max_size:]
        return file_list

    def add_unlabeled_data(self, fake, labels):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fake_path = os.path.join(self.unlabeled_data_dir, f"unlabeled_{timestamp}_fake.pt")
        labels_path = os.path.join(self.unlabeled_data_dir, f"unlabeled_{timestamp}_labels.pt")
        torch.save(fake.cpu(), fake_path)
        torch.save(labels.cpu(), labels_path)
        self.unlabeled_files.append((fake_path, labels_path))
        self.unlabeled_files = self._clean_old_files(self.unlabeled_files, self.unlabeled_data_dir)

    def add_labeled_data(self, fake, labels):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fake_path = os.path.join(self.labeled_data_dir, f"labeled_{timestamp}_fake.pt")
        labels_path = os.path.join(self.labeled_data_dir, f"labeled_{timestamp}_labels.pt")
        torch.save(fake.cpu(), fake_path)
        torch.save(labels.cpu(), labels_path)
        self.labeled_files.append((fake_path, labels_path))
        self.labeled_files = self._clean_old_files(self.labeled_files, self.labeled_data_dir)

    def _load_data(self, file_list, dir_path):
        fakes = []
        labels = []
        for f_path, l_path in file_list:
            if os.path.exists(f_path) and os.path.exists(l_path):
                fakes.append(torch.load(f_path))
                labels.append(torch.load(l_path))
        return torch.cat(fakes, dim=0) if fakes else None, torch.cat(labels, dim=0) if labels else None

    def sample_data(self, stage, current_model):
        unlabeled_fake, unlabeled_labels = self._load_data(self.unlabeled_files, self.unlabeled_data_dir)
        labeled_fake, labeled_labels = self._load_data(self.labeled_files, self.labeled_data_dir)

        if stage == 0:
            if unlabeled_fake is None or unlabeled_labels is None:
                return None, None
            total_unlabeled = unlabeled_fake.size(0)
            num_to_sample = min(args.batch_size * 5, total_unlabeled)
            indices = torch.randperm(total_unlabeled)[:num_to_sample]
            sampled_unlabeled_fake = unlabeled_fake[indices]
            sampled_unlabeled_labels = unlabeled_labels[indices]
            k = args.batch_size
            selected_unlabeled_fake, selected_unlabeled_labels = active_learning(sampled_unlabeled_fake, sampled_unlabeled_labels, current_model, k=k)
            return selected_unlabeled_fake, selected_unlabeled_labels
        else:
            if unlabeled_fake is None or unlabeled_labels is None or labeled_fake is None or labeled_labels is None:
                return None, None
            total_unlabeled = unlabeled_fake.size(0)
            num_to_sample_unlabeled = min(args.batch_size * 5, total_unlabeled)
            indices_unlabeled = torch.randperm(total_unlabeled)[:num_to_sample_unlabeled]
            sampled_unlabeled_fake = unlabeled_fake[indices_unlabeled]
            sampled_unlabeled_labels = unlabeled_labels[indices_unlabeled]
            k1 = int(args.batch_size * 0.7)
            selected_unlabeled_fake_1, selected_unlabeled_labels_1 = active_learning(sampled_unlabeled_fake, sampled_unlabeled_labels, current_model, k=k1)

            total_labeled = labeled_fake.size(0)
            num_to_sample_labeled = min(args.batch_size * 5, total_labeled)
            indices_labeled = torch.randperm(total_labeled)[:num_to_sample_labeled]
            sampled_labeled_fake = labeled_fake[indices_labeled]
            sampled_labeled_labels = labeled_labels[indices_labeled]
            k2 = int(args.batch_size * 0.3)
            selected_labeled_fake_2, selected_labeled_labels_2 = active_learning(sampled_labeled_fake, sampled_labeled_labels, current_model, k=k2)

            final_fake = torch.cat([selected_unlabeled_fake_1, selected_labeled_fake_2], dim=0)
            final_labels = torch.cat([selected_unlabeled_labels_1, selected_labeled_labels_2], dim=0)
            return final_fake, final_labels

    def _sample_batch(self, data, labels, batch_size):
        if data is None or labels is None:
            return None, None
        total = data.size(0)
        if total == 0:
            return None, None
        indices = torch.randperm(total)[:batch_size]
        return data[indices], labels[indices]

def active_learning(fake, labels, model, k=192, target_percentage=0.7):
    model.eval()
    with torch.no_grad():
        logits = model(fake)
        probs = F.softmax(logits, dim=1)
        confidence, _ = torch.max(probs, dim=1)
        num_top_samples = int(len(confidence) * target_percentage)
        _, top_confidence_indices = torch.topk(confidence, k=num_top_samples, largest=True)
        top_confidence_indices = top_confidence_indices.to(fake.device)
        top_confidence_fake = fake[top_confidence_indices]
        top_confidence_labels = labels[top_confidence_indices]
        top_confidence_entropy = -torch.sum(probs[top_confidence_indices] * torch.log(probs[top_confidence_indices] + 1e-10), dim=1)
        _, entropy_indices = torch.topk(top_confidence_entropy, k=min(k, len(top_confidence_entropy)), largest=True)
        entropy_indices = entropy_indices.to(top_confidence_indices.device)
        final_indices = top_confidence_indices[entropy_indices]
        if len(final_indices) < k:
            all_indices = torch.arange(len(fake), device=fake.device)
            non_selected_indices = all_indices[~torch.isin(all_indices, final_indices)]
            if len(non_selected_indices) > 0:
                remaining_indices = torch.randperm(len(non_selected_indices))[:k - len(final_indices)]
                final_indices = torch.cat([final_indices, non_selected_indices[remaining_indices]])
        selected_fake = fake[final_indices]
        selected_labels = labels[final_indices]
    return selected_fake, selected_labels

def data_generation(args, generator, clone_model_1, clone_model_2, target_model, epoch):
    generator.train()
    clone_model_1.eval()
    clone_model_2.eval()
    target_model.eval()

    best_fake = None
    best_loss = 1e6
    selected_model = 1  # 默认初始选择模型1

    z = torch.randn(size=(args.batch_size, args.gen_dim_z)).to(device)
    z.requires_grad = True

    optimizer_G = torch.optim.Adam(
        [{"params": generator.parameters()}, {"params": [z], "lr": args.lr_z}],
        lr=args.lr_G,
        betas=[0.5, 0.999],
    )

    pseudo_y = torch.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,)).to(device)
    soft_labels = L.smooth_one_hot(pseudo_y, classes=NUM_CLASSES, smoothing=args.label_smooth_factor)

    for step in range(args.N_G):
        fake = generator(z)
        aug_fake = std_aug(fake)

        logits_1 = clone_model_1(aug_fake)
        logits_2 = clone_model_2(aug_fake)

        loss_cls_1 = L.cross_entropy(logits_1, soft_labels)
        loss_cls_2 = L.cross_entropy(logits_2, soft_labels)
        if loss_cls_1 >= loss_cls_2:
            current_loss_cls = loss_cls_1
            selected_model = 1
        else:
            current_loss_cls = loss_cls_2
            selected_model = 2

        loss_div_1 = L.div_loss(logits_1)
        loss_div_2 = L.div_loss(logits_2)
        if loss_div_1 >= loss_div_2:
            current_loss_div = loss_div_1
        else:
            current_loss_div = loss_div_2

        loss = current_loss_cls + current_loss_div * 3

        with torch.no_grad():
            if best_loss > loss.item() or best_fake is None:
                best_loss = loss.item()
                best_fake = fake

        optimizer_G.zero_grad()
        loss.backward()
        optimizer_G.step()

    pseudo_labels = target_model(best_fake).topk(1, 1)[1].reshape(-1)
    return best_fake, pseudo_labels, selected_model

def train_clone_model(args, clone_model_1, clone_model_2, target_model,
                      optimizer_1, optimizer_2, epoch, data_storage, stage=0, selected_model=1):
    global curr_query_times
    target_model.eval()
    clone_model_1.train()
    clone_model_2.train()

    best_loss = float('inf')
    best_fake_hee = None
    best_hard_labels = None

    current_model = clone_model_1 if selected_model == 1 else clone_model_2

    for step in range(args.N_C):
        train_fake, train_labels = data_storage.sample_data(stage, current_model)
        if train_fake is None:
            logger.warning(f"Step {step} of Epoch {epoch}: No data to sample, skip.")
            continue

        fake, labels = train_fake.to(device), train_labels.to(device)
        aug_fake = std_aug(fake)

        if args.query_mode == "HEE":
            fake_hee = generate_hee(args, current_model,strong_aug(aug_fake))
            logits_T = target_model(fake_hee).detach()
            hard_labels = logits_T.topk(1, 1)[1].reshape(-1)

            logits_1 = clone_model_1(fake_hee)
            loss_1 = F.cross_entropy(logits_1, hard_labels)
            logits_2 = clone_model_2(fake_hee)
            loss_2 = F.cross_entropy(logits_2, hard_labels)

            curr_query_times += fake_hee.size(0)
            current_loss = max(loss_1.item(), loss_2.item())

            if current_loss < best_loss:
                best_loss = current_loss
                best_fake_hee = fake_hee.detach()
                best_hard_labels = hard_labels.detach()

            optimizer_1.zero_grad()
            loss_1.backward()
            optimizer_1.step()

            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

    if best_fake_hee is not None and best_hard_labels is not None:
        data_storage.add_labeled_data(best_fake_hee, best_hard_labels)

def main():
    global best_nature_acc_1, best_robust_acc_1, best_nature_acc_2, best_robust_acc_2
    best_nature_acc_1 = 0
    best_robust_acc_1 = 0
    best_nature_acc_2 = 0
    best_robust_acc_2 = 0
    logger.info(args)

    data_storage = FileDataStorage(max_size=5000)

    testset = getattr(datasets, args.data)(
        root=args.data_path, train=False, download=True, transform=transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size, shuffle=False
    )

    clone_model_1 = getattr(models, args.arch)(num_classes=NUM_CLASSES)
    clone_model_1 = nn.DataParallel(clone_model_1).to(device)

    clone_model_2 = getattr(models, 'ResNet34')(num_classes=NUM_CLASSES)
    clone_model_2 = nn.DataParallel(clone_model_2).to(device)

    target_model = getattr(models, args.target_arch)(num_classes=NUM_CLASSES)
    target_model = nn.DataParallel(target_model).to(device)
    state_dict = torch.load(target_path, map_location=device)
    target_model.load_state_dict(state_dict["model_state_dict"])
    target_model.eval()

    generator = models.Generator(nz=args.gen_dim_z, ngf=64, img_size=img_size, nc=nc)
    generator = nn.DataParallel(generator).to(device)

    optimizer_1 = torch.optim.SGD(
        clone_model_1.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    optimizer_2 = torch.optim.SGD(
        clone_model_2.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_1, args.epochs, eta_min=2e-4
    )
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_2, args.epochs, eta_min=2e-4
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        fake, labels, selected_model = data_generation(args, generator, clone_model_1, clone_model_2, target_model, epoch)
        data_storage.add_unlabeled_data(fake, labels)

        stage = 0 if epoch <= args.epochs * 0.5 else 1

        train_clone_model(
            args, clone_model_1, clone_model_2, target_model,
            optimizer_1, optimizer_2, epoch, data_storage,
            stage=stage, selected_model=selected_model
        )

        scheduler_1.step()
        scheduler_2.step()

        nature_acc_1 = clean_test(clone_model_1, test_loader)
        robust_acc_1 = adv_test(clone_model_1, test_loader)

        nature_acc_2 = clean_test(clone_model_2, test_loader)
        robust_acc_2 = adv_test(clone_model_2, test_loader)

        epoch_time = time.time() - start_time

        logger.info(
            f"Epoch {epoch} Finish, Time Cost {epoch_time:.2f}s, "
            f"Clone Model 1 Nature Acc {nature_acc_1:.4f}, Robust Acc {robust_acc_1:.4f}, "
            f"Clone Model 2 Nature Acc {nature_acc_2:.4f}, Robust Acc {robust_acc_2:.4f}"
        )

        is_best_robust_1 = robust_acc_1 > best_robust_acc_1
        best_robust_acc_1 = max(robust_acc_1, best_robust_acc_1)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": clone_model_1.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                "nature_acc": float(nature_acc_1),
                "robust_acc": float(robust_acc_1),
            },
            epoch,
            is_best_robust_1,
            "robust_1",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

        is_best_nature_1 = nature_acc_1 > best_nature_acc_1
        best_nature_acc_1 = max(nature_acc_1, best_nature_acc_1)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": clone_model_1.state_dict(),
                "optimizer": optimizer_1.state_dict(),
                "nature_acc": float(nature_acc_1),
                "robust_acc": float(robust_acc_1),
            },
            epoch,
            is_best_nature_1,
            "nature_1",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

        is_best_robust_2 = robust_acc_2 > best_robust_acc_2
        best_robust_acc_2 = max(robust_acc_2, best_robust_acc_2)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": clone_model_2.state_dict(),
                "optimizer": optimizer_2.state_dict(),
                "nature_acc": float(nature_acc_2),
                "robust_acc": float(robust_acc_2),
            },
            epoch,
            is_best_robust_2,
            "robust_2",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

        is_best_nature_2 = nature_acc_2 > best_nature_acc_2
        best_nature_acc_2 = max(nature_acc_2, best_nature_acc_2)
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": clone_model_2.state_dict(),
                "optimizer": optimizer_2.state_dict(),
                "nature_acc": float(nature_acc_2),
                "robust_acc": float(robust_acc_2),
            },
            epoch,
            is_best_nature_2,
            "nature_2",
            save_path=checkpoint_path,
            save_freq=args.save_freq,
        )

    logger.info("Best Nature ACC 1: %.4f", best_nature_acc_1)
    logger.info("Best Robust ACC 1: %.4f", best_robust_acc_1)
    logger.info("Best Nature ACC 2: %.4f", best_nature_acc_2)
    logger.info("Best Robust ACC 2: %.4f", best_robust_acc_2)

    logger.info("Evaluation Results for Clone Model 1:")
    best_robust_model_1 = getattr(models, args.arch)(num_classes=NUM_CLASSES)
    best_robust_model_1 = nn.DataParallel(best_robust_model_1).to(device)
    best_robust_model_1.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "best_robust_1_checkpoint.tar"))["model_state_dict"]
    )
    best_robust_model_1.eval()
    eval_results_1 = robust_eval(best_robust_model_1, test_loader, device)
    logger.info(eval_results_1)

    logger.info("Evaluation Results for Clone Model 2:")
    best_robust_model_2 = getattr(models, 'ResNet34')(num_classes=NUM_CLASSES)
    best_robust_model_2 = nn.DataParallel(best_robust_model_2).to(device)
    best_robust_model_2.load_state_dict(
        torch.load(os.path.join(checkpoint_path, "best_robust_2_checkpoint.tar"))["model_state_dict"]
    )
    best_robust_model_2.eval()
    eval_results_2 = robust_eval(best_robust_model_2, test_loader, device)
    logger.info(eval_results_2)

if __name__ == "__main__":
    main()