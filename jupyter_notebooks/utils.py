import torch
import torch.nn as nn
from nano_gpt import GPT2Model, GPT2Config
from tqdm import tqdm
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from scripts.tasks import LinearRegression
import warnings
from sklearn import tree
#import xgboost as xgb

import os


relevant_model_names = {
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
}


def eval_unlooped_model(model, xs, ys, add_inputs_embeds=False):
    """

    :param model:
    :param xs: [N, n, d]
    :param ys: [N, n]
    :return: err: [N, n]
    """
    sample_size = xs.shape[0]
    n_points = xs.shape[1]
    batch_size = 128
    assert sample_size % batch_size == 0
    with torch.no_grad():
        y_pred_total = torch.zeros(sample_size, n_points)
        for batch_idx in range(sample_size // batch_size):
            xs_train = xs[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            ys_train = ys[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            y_pred = model(xs_train, ys_train, add_inputs_embeds=add_inputs_embeds).detach()
            y_pred_total[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred
        err = (y_pred_total - ys.cpu()).square()
    return err, y_pred_total


def eval_looped_model(model, xs, ys, loop_max):
    """

    :param model:
    :param xs: [N, n, d]
    :param ys: [N, n]
    :param loop_max: positive integer, indicating the maximum number of loops to apply
    :return:
        err: [N, n]
        loop_err: [N, loop_max]
    """
    sample_size = xs.shape[0]
    n_points = xs.shape[1]
    batch_size = 128
    assert sample_size % batch_size == 0
    with torch.no_grad():
        y_pred_total = torch.zeros(sample_size, n_points)  # [N, n]
        y_pred_last = torch.zeros(sample_size, loop_max)  # [N, T]
        for batch_idx in range(sample_size // batch_size):
            xs_train = xs[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            ys_train = ys[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            y_pred_list = model(xs_train, ys_train, 0, loop_max)  # list of [B, n], length T
            if len(y_pred_list) == 1:
                y_pred_list = y_pred_list[0]

            y_pred_total[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred_list[-1].detach()
            tmp_list = [y_pred[:, [-1]] for y_pred in y_pred_list]  # list of [B, 1], length T
            tmp_arry = torch.cat(tmp_list, dim=1)  # [B, T]
            y_pred_last[batch_idx * batch_size: (batch_idx + 1) * batch_size] = tmp_arry
        err = (y_pred_total - ys.cpu()).square()  # [n,]
        loop_err = (y_pred_last - ys.cpu()[:, [-1]]).square()  # [N, T] - [N, 1]
    return err, loop_err


def get_model(model, result_dir, run_id, step, best=False, strict=True):
    if best:
        model_path = os.path.join(result_dir, run_id, 'model_best.pt')
        state_dict = torch.load(model_path, map_location='cpu')['state_dict']
        best_err = torch.load(model_path, map_location='cpu')['loss']
        print("saved model with loss:", best_err)
    if step == -1:
        model_path = os.path.join(result_dir, run_id, 'state.pt')
        state_dict = torch.load(model_path, map_location='cpu')['model_state_dict']
    else:
        model_path = os.path.join(result_dir, run_id, 'model_{}.pt'.format(step))
        state_dict = torch.load(model_path, map_location='cpu')['model']
    
#     return state_dict
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=strict)
    
    return model


def aggregate_metrics(result_dict, d, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    for model_name in result_dict.keys():
        errs = result_dict[model_name]
        tmp = {}
        tmp["mean"] = errs.mean(0) / d
        tmp["std"] = errs.std(0, unbiased=True) / d
        n = len(errs)
        bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
        bootstrap_means = errs[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
        tmp["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :] / d
        tmp["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :] / d
        results[model_name] = tmp

    return results


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):
        # should we be picking k optimally
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"{n_neighbors}-Nearest Neighbors"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"Least Squares"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


class AveragingModel:
    def __init__(self):
        self.name = "Averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"Lasso (alpha={alpha})"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class RidgeRegressionModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"Ridge (alpha={alpha})"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Ridge(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        warnings.filterwarnings("ignore", category=DeprecationWarning)
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"ridge regression convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)

    
class NeuralNetwork(nn.Module):
    def __init__(self, in_size=50, hidden_size=1000, out_size=1):
        super(NeuralNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ParallelNetworks(nn.Module):
    def __init__(self, num_models, model_class, **model_class_init_args):
        super(ParallelNetworks, self).__init__()
        self.nets = nn.ModuleList(
            [model_class(**model_class_init_args) for i in range(num_models)]
        )

    def forward(self, xs):
        assert xs.shape[0] == len(self.nets)

        for i in range(len(self.nets)):
            out = self.nets[i](xs[i])
            if i == 0:
                outs = torch.zeros(
                    [len(self.nets)] + list(out.shape), device=out.device
                )
            outs[i] = out
        return outs


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        # self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"
        self.name = f"gd_model_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}"

    def __call__(self, xs, ys, device=torch.device("cuda:0"), inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.to(device), ys.to(device)

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            # model = torch.compile(model)
            model.to(device)
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in tqdm(range(self.num_steps)):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self, max_depth=None, learning_rate=None, n_estimators=100):
        self.name = f"xgboost_depth_{max_depth}_lr_{learning_rate}_n_estimators_{n_estimators}"
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor(
                        max_depth=self.max_depth, learning_rate=self.learning_rate, n_estimators=self.n_estimators)

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
    
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
    

def similarity_sort(xs, ys):
    idx = np.argsort(torch.cosine_similarity(xs[:,:-1,:], xs[:, [-1], :], dim=-1))
    idx_expanded = idx.unsqueeze(2).expand(-1, -1, xs[:,:-1,:].size(2))

    sorted_xs = torch.gather(xs[:,:-1,:], 1, idx_expanded)
    new_xs = torch.cat((sorted_xs, xs[:,[-1],:]), dim=1)

    sorted_ys = torch.gather(ys[:,:-1], 1, idx)
    new_ys = torch.cat((sorted_ys, ys[:,[-1]]), dim=1)

    return new_xs, new_ys

def create_repeated_sequence(d, k, n):
    real_task = LinearRegression(batch_size=1, 
                                 n_points=k, 
                                 n_dims=d, 
                                 n_dims_truncated=d, 
                                 device='cpu')
    xs, ys = real_task.xs, real_task.ys
    xs = xs.squeeze(0).repeat(n, 1).unsqueeze(0)
    ys = ys.squeeze(0).repeat(n, 1).unsqueeze(0)
    return xs, ys

def get_attention_weights(inputs, model):
    batch_size, seq_length, embed_dim = inputs.size()
    num_heads = model.configuration.n_head
    
    qkv = model._backbone.transformer.h[0].attn.c_attn(inputs)  # (batch_size, sequence_length, 3 * embed_dim)
    
    head_dim = embed_dim // num_heads
    qkv = qkv.view(batch_size, seq_length, 3, num_heads, head_dim)
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # (B, h, n, n)
    
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).to(inputs.device)
    attn_scores = attn_scores.masked_fill(mask == 1, float('-inf'))
    
    attn_weights = nn.functional.softmax(attn_scores, dim=-1)
    
    return attn_weights


def get_prefix_matching_scores(A, k, n):
    mean_values = []
    A_new = A[::2, 1::2] #rows are xs, columns are ys
    for i in range(len(A_new)):
        x_number = i % k
        mean_values += [A_new[i, x_number::k].mean().item()]

    final_mean_values = []
    for i in range(k):
        final_mean_values += [np.mean(mean_values[i::k])]

    return final_mean_values


def get_prefix_matching_score(A, k, n):
    A_new = A[::2, 1::2]
    self_attentions = []
    for i in range(len(A_new)):
        x_number = i % k
        self_attentions += [A_new[i, x_number::k]]
    final_scores = []
    for i in range(k):
        rows = self_attentions[i::k]
        rows = torch.hstack([row[row > 0] for row in rows])
        final_scores += [rows.mean()]
    return np.mean(final_scores)

def prefix_matching(model, k, n, max_loops=20):
    """Generates a sequence (x_i, f(x_i)) of length k repeated n times
    returns a matrix of size (max_loops+1, n_head) where (i,j) values
    shows the prefix matching score of the specific head: average attention score
    of x_i to all the previous y_i
    """
    xs, ys = create_repeated_sequence(d=20, k=k, n=n)
    _, hidden_states = model(xs, ys, 0, max_loops, return_hidden_states=True)
    embeds = model._read_in(combine(xs, ys))
    hidden_states = [embeds] + hidden_states
    all_attentions = []
    prefix_matching_scores = np.zeros((max_loops+1, model.configuration.n_head))
    for i, x in enumerate(hidden_states):
        attn = get_attention_weights(x, model).squeeze(0)
        all_attentions += [attn]
        for j, head in enumerate(attn):
            prefix_matching_scores[i, j] = get_prefix_matching_score(head.detach(), k, n)

    return prefix_matching_scores