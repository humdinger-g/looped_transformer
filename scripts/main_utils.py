import torch
import datetime
import uuid
import os
import numpy as np
from torch.utils.data import Dataset


class my_Dataset(Dataset):
    """This function reads the data from a pickle file and creates a PyTorch dataset, which contains:
    state, action, reward, reward-to-go, target
    """
    def __init__(self, xs, ys):
        self.xs = xs  # [N, n, d]
        self.ys = ys  # [N, n]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, index):
        return {
            'x': self.xs[index].float(),  # [n, d]
            'y': self.ys[index].float()
        }


def gen_dataloader(task_sampler, num_sample, batch_size):
    from torch.utils.data import DataLoader

    xs_list, ys_list = [], []
    for i in range(num_sample // batch_size):
        task = task_sampler()
        xs, ys = task.xs.float().cpu(), task.ys.float().cpu()
        xs_list.extend(xs)
        ys_list.extend(ys)
    dataset = my_Dataset(xs_list, ys_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def init_device(args):
    cuda = args.gpu.cuda
    gpu = args.gpu.n_gpu
    if cuda:
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")
        torch.set_num_threads(4)
    return device


def rm_orig_mod(state_dict):
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    return state_dict


def load_pretrained_model(args, model, optimizer, curriculum, device):
    state_path = os.path.join(args.out_dir, "state.pt")
    starting_step = 0
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)  # NOTE: change to cpu if OOM
        state_dict = state["model_state_dict"]  # rm_orig_mod(state["model_state_dict"])
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()
        del state
        del state_dict
    elif args.model.pretrained_path is not None:
        state = torch.load(args.model.pretrained_path, map_location=device)
        if "model_state_dict" in state.keys():
            state_dict = state["model_state_dict"]  # rm_orig_mod(state["model_state_dict"])
            model.load_state_dict(state_dict, strict=False)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            for i in range(state["train_step"] + 1):
                curriculum.update()
            starting_step = state["train_step"]
            del state
            del state_dict
        else:
            state_dict = rm_orig_mod(state["model"])
            model.load_state_dict(state_dict)

            def find_train_step(s):
                step = s[s.find('model_') + 6:s.find('.pt')]
                return int(step)

            num_train_step = find_train_step(args.model.pretrained_path)
            starting_step = num_train_step
            for i in range(num_train_step + 1):
                curriculum.update()
            del state
            del state_dict
    else:
        print("train from scratch")
    return args, model, optimizer, curriculum, state_path, starting_step


def get_run_id(args):
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    run_id = "{}-{}-".format(now, args.wandb.name) + str(uuid.uuid4())[:4]
    return run_id


def combine(xs_b, ys_b):
    """
    :param xs_b: shape [B, n, d_in]
    :param ys_b: shape [B, n]
    :return: shape [B, 2n, d_in + 1]
    """
    freq = 2
    B, n, d = xs_b.shape
    device = xs_b.device

    ys_b_wide = torch.cat(
        (
            ys_b.view(B, n, 1),
            torch.zeros(B, n, d-1, device=device),
        ),
        axis=2,
    )

    zs = torch.stack((xs_b, ys_b_wide), dim=2)
    zs = zs.view(B, freq * n, d)

    return zs


def sample_pairs_separate(xs: torch.Tensor,
                 ys: torch.Tensor,
                 alpha: float = 0.5,
                 last: bool = False) -> torch.Tensor:
    """
    takes input prompt P of pairs (x, y) and leaves
    random ratio alpha of them

    Args:
        xs (torch.Tensor): tensor of x values of shape [B, n, d]
        ys (torch.Tensor): tensor of y values of shape [B, n, d]
        alpha (float): ratio of pairs to sample. Defaults to 0.5
        last (bool): if True, samlpes last examples. Defaults to False

    Returns:
        torch.Tensor: samples from input embeds of shape [B, 2*int(n*alpha), d]
    """
    B, n, d = xs.shape
    num = n // 2
    if last:
        idx = np.arange(num)[-int(num*alpha):]
    else:
        idx = np.random.choice(num, size=int(num*alpha), replace=False)

    new_xs = xs[:,idx,:]
    new_ys = ys[:,idx]

    return new_xs, new_ys


def sample_inputs(embeds: torch.Tensor,
                 alpha: float = 0.5,
                 last: bool = False) -> torch.Tensor:
    """
    takes input prompt P of pairs (x, y) and leaves
    random ratio alpha of them

    Args:
        embeds (torch.Tensor): input tensor of shape [B, 2n, d]
        alpha (float): ratio of pairs to sample. Defaults to 0.5
        last (bool): if True, samlpes last examples. Defaults to False

    Returns:
        torch.Tensor: samples from input embeds of shape [B, 2*int(n*alpha), d]
    """
    B, n, d = embeds.shape
    num = n // 2
    if last:
        idx = np.arange(num)[-int(num*alpha):]
    else:
        idx = np.random.choice(num, size=int(num*alpha), replace=False)

    # new_xs = embeds[:,0::2,:][:,idx,:]
    # new_ys = embeds[:,1::2,:][:,idx,:]

    # new_embeds = torch.zeros(B, int(2*num*alpha), d)
    # new_embeds[:,0::2,:] = new_xs
    # new_embeds[:,1::2,:] = new_ys
    mask = torch.zeros_like(embeds)
    mask[:, 2*idx, :] = 1
    mask[:, 2*idx+1, :] = 1
    mask = mask != 1
    return torch.masked_fill(embeds, mask, 0)


def iterative_regression(xs, ys, lr=0.001, epochs=30):
    weights, outs = [], []
    w = torch.randn((xs.shape[1], 1)).to(xs.device)
    for i in range(epochs):
        w -= lr * xs.T @ (xs @ w - ys.unsqueeze(-1))
        outs += [(xs @ w).squeeze(-1)]
    
    err = (outs[-1] - ys.unsqueeze(0)).square().mean().item()

    return torch.stack(outs), err

def get_best_preds(xs, ys, LR=np.logspace(-5, -1, base=10, num=30), epochs=30):
    all_outs = []
    best_lrs = [] ###
    for i in range(xs.shape[0]):
        err = float('inf')
        outs = []
        for lr in LR:
            cur_outs, cur_err = iterative_regression(xs[i], ys[i], lr=lr, epochs=epochs)
            if cur_err < err:
                outs = cur_outs
                best_lr = lr

        all_outs += [outs]
        best_lrs += [best_lr]
    
    return torch.stack(all_outs), best_lrs

def similarity_sort(xs, ys):
    idx = np.argsort(torch.cosine_similarity(xs[:,:-1,:], xs[:, [-1], :], dim=-1))
    idx_expanded = idx.unsqueeze(2).expand(-1, -1, xs[:,:-1,:].size(2))

    sorted_xs = torch.gather(xs[:,:-1,:], 1, idx_expanded)
    new_xs = torch.cat((sorted_xs, xs[:,[-1],:]), dim=1)

    sorted_ys = torch.gather(ys[:,:-1], 1, idx)
    new_ys = torch.cat((sorted_ys, ys[:,[-1]]), dim=1)

    return new_xs, new_ys