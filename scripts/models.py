import torch
import torch.nn as nn
from nano_gpt import GPT2Model, GPT2Config, LayerNorm
from main_utils import sample_inputs
import numpy as np

MAX_NUM_CLASS = 2  # for openML classification task

def build_model(conf):
    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            minimal=conf.minimal,
            pred_type=conf.pred_type,
        )
    elif conf.family == 'gpt2_loop':
        model = TransformerModelLooped(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
            minimal=conf.minimal,
            loop_func=conf.loop_func,
            decay=conf.decay,
            pred_type=conf.pred_type,
        )
    elif conf.family == 'gpt2_tying':
        model = TransformerModelTying(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )
    elif conf.family == 'gpt2_loop_parallel':
        model = TransformerLoopedParallel(
            S=conf.S,
            L=conf.n_layer,
            config=conf
        )

    else:
        raise NotImplementedError

    return model


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, minimal=True, pred_type='regression'):

        super(TransformerModel, self).__init__()
        self.freq = 2
        self.ind = 0
        configuration = GPT2Config()
        configuration.block_size = self.freq * n_positions + 1
        configuration.n_layer = n_layer
        configuration.n_head = n_head
        configuration.n_embd = n_embd
        configuration.minimal = minimal
        configuration.dropout = 0.0
        configuration.bias = True
        configuration.dropout = 0.
        self.configuration = configuration

        self.n_positions = n_positions  # n = points in this setting
        self.n_dims = n_dims  # input dimension, d_in
        self.n_embd = n_embd  # d
        self.n_layer = n_layer
        self.minimal = minimal
        self._pred_type = pred_type

        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(self.configuration)
        if self._pred_type == 'regression':
            self._read_out = nn.Linear(n_embd, 1)
        elif self._pred_type == 'classification':
            self._read_out = nn.Linear(n_embd, MAX_NUM_CLASS)  # NOTE: hard-code

        self.print_flag = False

    def _combine(self, xs_b, ys_b):
        """
        :param xs_b: shape [B, n, d_in]
        :param ys_b: shape [B, n]
        :return: shape [B, 2n, d_in + 1]
        """
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
        zs = zs.view(B, self.freq * n, d)

        return zs

    def forward(self, xs, ys, add_inputs_embeds=True, return_attentions=False):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :return:
        """

        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]

        f_output = self._backbone(
            inputs_embeds=embeds, position_ids=None, rm_pos_embd=False, add_inputs_embeds=add_inputs_embeds,
            return_attn=return_attentions)  # [B, 2n, d]
        if return_attentions:
            f_output, attns = f_output
        
        prediction = self._read_out(f_output)  # [B, 2n, d] -> [B, 2n, 1]
        if self._pred_type == 'regression':
            y = prediction[:, self.ind::self.freq, 0]
        elif self._pred_type == 'classification':
            y = prediction[:, self.ind::self.freq]
        else:
            raise NotImplementedError

        return y if not return_attentions else (y, attns)


class TransformerModelTying(TransformerModel):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):

        super(TransformerModelTying, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head)

        self.configuration.n_layer = 1

        self._backbone = GPT2Model(self.configuration)

        self.print_flag = False

    def f(self, output):
        f_output = self._backbone(inputs_embeds=output)  # [B, 2n + 1, d]
        return f_output

    def forward(self, xs, ys, add_inputs_embeds):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :param n_loop_start: int
        :param n_loops: int
        :return:
        """
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        output = embeds  # also of shape [B, 2n, d]

        for idx in range(self.n_layer):
            output = self.f(output)
        prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
        y = prediction[:, self.ind::self.freq, 0]  # [B, n]

        return y


class TransformerModelLooped(TransformerModel):
    def __init__(
            self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4, loop_func='z=f(x+z)', minimal=True, decay='seq', pred_type='regression'):

        super(TransformerModelLooped, self).__init__(
            n_dims, n_positions, n_embd, n_layer, n_head, minimal, pred_type)
        self.loop_func = loop_func
        self.decay = decay
        if decay == 'seq':
            self.gammas = nn.Parameter(torch.ones(100))

        elif decay == 'const':
            self.gammas = nn.Parameter(torch.ones(1))
            

    def f(self, output, embeds, sample='full', alpha=0.5, return_attentions=False):
        if sample == 'last':
            embeds = sample_inputs(embeds, alpha=alpha, last=True)
        elif sample == 'random':
            embeds = sample_inputs(embeds, alpha=alpha)

        if self.loop_func == 'z=f(x+z)':
            f_output = self._backbone(inputs_embeds=output + embeds, return_attn=return_attentions)  # [B, 2n + 1, d]
        elif self.loop_func == 'z=f(x*z)':
            f_output = self._backbone(inputs_embeds=output * embeds, return_attn=return_attentions)  # [B, 2n + 1, d]
        else:
            raise NotImplementedError
        return f_output

    def forward(self, xs, ys, n_loop_start, n_loops, return_hidden_states=False, return_attentions=False):
        """
        :param xs: [B, n, d]
        :param ys: [B, n]
        :param n_loop_start: int
        :param n_loops: int
        :param return_hidden_states: bool = False -- if True, returns outputs of each
             transformer block (before self._read_out)
        :param return_attentions: bool = False -- if True, returns all attention maps
            shape is (max_loops, n_layer, batch_size, n_head, n, n)
        :return:

        """
        B, n, d_in = xs.shape
        zs = self._combine(xs, ys)  # [B, n, d_in], [B, n], [B, n] -> [B, 2n, d_in + 1]
        embeds = self._read_in(zs)  # [B, 2n, d_in + 1] -> [B, 2n, d]
        if self.loop_func in ['z=f(x+z)']:
            output = torch.zeros_like(embeds)  # also of shape [B, 2n, d]
        elif self.loop_func in ['z=f(x*z)']:
            output = torch.ones_like(embeds)  # also of shape [B, 2n, d]
        else:
            raise NotImplementedError("Currently we only support loop function z=f(x+z) or z=f(x*z).")

        pred_list = []
        hidden_states = []
        all_attentions = []
        for idx in range(n_loops):

            if self.decay == 'seq':
                gamma = self.gammas[idx]
            elif self.decay == 'const':
                gamma = self.gammas
            else:
                gamma = 1.0

            if idx < n_loop_start:  # this will save memory when n_loops large.
                with torch.no_grad():
                    if idx == 0:
                        output = self.f(output, embeds, return_attentions=return_attentions)
                    else:
                        output = self.f(output, gamma * embeds, return_attentions=return_attentions)
            else:
                if idx == 0:
                    output = self.f(output, embeds, return_attentions=return_attentions)
                else:
                    output = self.f(output, gamma * embeds, return_attentions=return_attentions)
                
                if return_attentions:
                    output, attentions = output
                    all_attentions += [torch.stack(attentions)]  

                prediction = self._read_out(output)  # [B, 2n, d] -> [B, 2n, 1]
                if self._pred_type == 'regression':
                    y = prediction[:, self.ind::self.freq, 0]
                elif self._pred_type == 'classification':
                    y = prediction[:, self.ind::self.freq]
                else:
                    raise NotImplementedError
                pred_list.append(y)
            if return_hidden_states:
                hidden_states += [output]
            if not self.print_flag:
                print(idx)
                self.print_flag = True

        out = (pred_list,)
        if return_hidden_states:
            out += (hidden_states,)
        if return_attentions:
            out += (torch.stack(all_attentions),)
        
        return out


class TransformerLoopedParallel(nn.Module):
    def __init__(self, S: int, L: int, config: GPT2Config):
        """initialization of a model with S parallel streams of L 
        decoder transformer models

        Args:
            S (int): number of parallel streams of looped layers
            L (int): number of layers in each stream
            config (GPT2Config): config for the remaining parameters
        """
        super().__init__() 
        self.S = S
        self.streams = nn.ModuleList([
            TransformerModelLooped(config['n_dims'],
                                   config['n_positions'],
                                   config['n_embd'], 
                                   L,
                                   config['n_head'],
                                   config['loop_func'])
        for _ in range(S)])

        self.weights = nn.Linear(S, 1, bias=False)


    def forward(self, xs, ys, n_loops_start, n_loops):
        stream_outputs = []
        # each stream output has shape [K, B, n], total S streams
        for s in range(self.S):
            stream_outputs += [torch.stack(self.streams[s](xs, ys, n_loops_start, n_loops)[0])]
            #stream_outputs += [torch.cat(self.streams[s](xs, ys, n_loops_start, n_loops), dim=0).unsqueeze(0)]

        # get weighted average of all the streams, [S, K, B, n] -> [K, B, n]
        res = self.weights(torch.stack(stream_outputs).permute(1,2,3,0))
        return res.squeeze(-1).squeeze(-1).unbind(0)