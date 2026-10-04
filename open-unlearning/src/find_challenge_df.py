import copy
import os
from collections import OrderedDict
import matplotlib.pyplot as plt

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import unlearn
import utils

import numpy as np
from trainer import validate
from torch.utils.data import DataLoader


def norm_grad(x, p):
    norm_value = torch.norm(x, p=p)
    return (x.abs()**(p - 1)) * x.sign() / norm_value**(p - 1)


def l1_regularization(model):
    params_vec = []
    for param in model.parameters():
        params_vec.append(param.view(-1))
    return torch.linalg.norm(torch.cat(params_vec), ord=1)

def update_w(
    origin_train_set, 
    w=None, 
    class_to_replace: int = None, 
    num_indexes_to_replace: int = None,
    seed: int = 1,
    batch_size:int = 128,
    shuffle: bool = False,
    only_mark: bool = True,
    args=None
):
    def replace_loader_dataset(
        dataset, batch_size=batch_size, seed=seed, shuffle=True
    ):
        setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )
    
    train_full_loader = replace_loader_dataset(
        origin_train_set, batch_size=batch_size, seed=seed, shuffle=shuffle
    )

    train_set = copy.deepcopy(origin_train_set)
    if w is None:
        if class_to_replace is not None:
            indexes = replace_class(
                train_set,
                class_to_replace,
                num_indexes_to_replace=num_indexes_to_replace,
                seed=seed - 1,
                only_mark=only_mark,
            )

            # binary
            w = torch.zeros(len(train_set))
            w[indexes] = 1

            # uniform
            # w = torch.ones(len(train_set)) / len(train_set)

    else:
        indexes = torch.where(w == 1)[0].tolist()
        replace_indexes(train_set, indexes, seed, only_mark)

    if args.dataset == "cifar10" or args.dataset == "cifar100" or args.dataset == "trans_cifar10":
        forget_dataset = copy.deepcopy(train_set)
        marked = forget_dataset.targets < 0
        forget_dataset.data = forget_dataset.data[marked]
        forget_dataset.targets = -forget_dataset.targets[marked] - 1
        forget_loader = replace_loader_dataset(
            forget_dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )

        retain_dataset = copy.deepcopy(train_set)
        marked = retain_dataset.targets >= 0
        retain_dataset.data = retain_dataset.data[marked]
        retain_dataset.targets = retain_dataset.targets[marked]
        retain_loader = replace_loader_dataset(
            retain_dataset, batch_size=batch_size, seed=seed, shuffle=shuffle
        )

    return train_full_loader, forget_loader, retain_loader, w,


def main():

    gaps = {"UA": []}
    w_records = []
    bi_w_records = []
    w_norms = []
    bi_w_norms = []
    select_epoch_losses = []

    # binary
    w = torch.zeros(len(train_set))
    w[indexes] = 1

    (
        model,
        train_set, 
        valid_set, 
        test_set
    ) = utils.setup_model_indexdataset(args)
    model.cuda()


    criterion = nn.CrossEntropyLoss(reduction="none")
    train_full_loader, forget_loader, remain_loader, w = utils.update_w(train_set,
                                                    class_to_replace=args.class_to_replace, 
                                                    num_indexes_to_replace=args.num_indexes_to_replace,
                                                    seed=args.seed,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    args=args
                                                    )
    w = w.cuda()
    evaluation_result = None

    for epoch in range(args.select_epochs):

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        if args.unlearn == "w_RL" or args.unlearn == "w_boundary_shrink" or args.unlearn == "w_boundary_expanding" or args.unlearn == "w_scrub":
            pre_data_loaders = OrderedDict(
                retain=remain_loader, forget=forget_loader, val=val_loader, test=test_loader
            )
            unlearn_method(pre_data_loaders, model, criterion, args, w, mask)

        else:
            unlearn_method(train_full_loader, model, criterion, args, w, mask)

        if args.mode == "optm":
            w, select_epoch_loss = unlearn.optimize_select(train_full_loader, model, criterion, args, w)
        elif args.mode == "re_optm":
            w, select_epoch_loss = unlearn.reverse_optimize_select(train_full_loader, model, criterion, args, w)

        select_epoch_losses.append(select_epoch_loss.item())

        topk_indices = torch.topk(w, args.num_indexes_to_replace)[1]
        bi_w = torch.zeros_like(w)
        bi_w[topk_indices] = 1

        w_records.append(w.cpu().numpy())
        bi_w_records.append(bi_w.cpu().numpy())

        _, forget_loader, remain_loader, _ = utils.update_w(train_set,
                                    w=bi_w,
                                    class_to_replace=args.class_to_replace, 
                                    seed=args.seed,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    args=args)

        if (epoch + 1) % args.feq_to_bi == 0:
            w = bi_w
            if epoch == args.select_epochs - 1:
                _, forget_loader, remain_loader, _ = utils.update_w(train_set,
                                                                    w=w,
                                                                    class_to_replace=args.class_to_replace, 
                                                                    seed=args.seed,
                                                                    batch_size=args.batch_size,
                                                                    shuffle=True,
                                                                    args=args)

    w_path = os.path.join(args.save_dir, "select_weight.pth.tar")
    gaps['w'] = w_records
    gaps['bi_w'] = bi_w_records
    gaps['loss'] = select_epoch_losses
    torch.save(gaps, w_path)


if __name__ == "__main__":
    main()