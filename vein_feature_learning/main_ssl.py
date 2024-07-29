import torch
from models.models import ResNets
from models.models import MLPHead
from trainer import BYOLTrainer, SimCLRTrainer
import numpy as np
import random
from data.dataset import get_transforms_ssl, MultiViewDataInjector, VeinDataset
from torch.utils.data.dataloader import DataLoader
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='FVUSM', help='name of the dataset')
    parser.add_argument('--network', type=str, default='resnet18', help='name of the network: {resnet18, resnet34, resnet50}')
    parser.add_argument('--loss', type=str, default='byol', help="self-supervised learning method: {simclr, byol}")
    parser.add_argument('--lr', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='learning rate decay factor')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum factor for sgd optimizer')
    parser.add_argument('--wd', type=float, default=4e-4, help='weight decay factor')
    parser.add_argument('--encoder_momentum', type=float, default=0.996, help='target encoder momentum for byol')
    parser.add_argument('--temperature', type=float, default=0.05, help='temperature factor for simclr')
    parser.add_argument('--max_epoch', type=int, default=80, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--seed', type=int, default=99, help='random seed for repeating results')
    parser.add_argument('--synthetic_num', type=int, default=10000, help='the number of synthetic samples used for training')
    parser.add_argument('--trainset', type=str, help='train set path')
    parser.add_argument('--testset', type=str, help='test set path')
    parser.add_argument("--save_image", action='store_true', help="save the augmented images during training")
    parser.add_argument('--simple_eval', action='store_true', help="whether to use simplified evaluation protocol")
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    if args.dataset_name.lower() == "fvusm":
        sample_per_class = 12
    elif args.dataset_name.lower() == "palmvein":
        sample_per_class = 20
    else:
        raise ValueError("Dataset not supported!")

    data_transform_train, data_transform_test = get_transforms_ssl(args.dataset_name)
    train_dataset = VeinDataset(root=args.trainset, transform=MultiViewDataInjector([data_transform_train, data_transform_train]), num_samples=args.synthetic_num, sample_per_class=1)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, drop_last=False, shuffle=True, pin_memory=True)

    test_dataset = VeinDataset(root=args.testset, sample_per_class=sample_per_class, transform=data_transform_test)
    testloader = DataLoader(test_dataset, batch_size=64, num_workers=4, drop_last=False, shuffle=False, pin_memory=True)

    if args.loss == "byol":
        # online network
        online_network = ResNets(backbone=args.network, head_type='byol').to(device)
        # predictor network
        predictor = MLPHead(in_channels=128, mlp_hidden_size=512, projection_size=128).to(device)
        # target encoder
        target_network = ResNets(backbone=args.network, head_type='byol').to(device)
        optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=args.lr_decay)
        trainer = BYOLTrainer(online_network=online_network,
                              target_network=target_network,
                              optimizer=optimizer,
                              predictor=predictor,
                              device=device,
                              m=args.encoder_momentum,
                              max_epochs=args.max_epoch,
                              lr_scheduler=lr_scheduler,
                              batch_size=args.batch_size,
                              save_image=args.save_image,
                              args=args)
    elif args.loss == "simclr":
        # online network
        online_network = ResNets(backbone=args.network, head_type='simclr').to(device)
        optimizer = torch.optim.SGD(online_network.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=args.lr_decay)
        trainer = SimCLRTrainer(online_network=online_network,
                                optimizer=optimizer,
                                device=device,
                                max_epochs=args.max_epoch,
                                lr_scheduler=lr_scheduler,
                                temperature=args.temperature,
                                batch_size=args.batch_size,
                                save_image=args.save_image,
                                args=args)
    else:
        raise ValueError(f"Invalid option {args.loss}")

    trainer.train(trainloader, testloader)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    set_seed(args.seed)
    main()
