# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
from dataclasses import dataclass
from typing import Optional, List

import sys, os

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
# 向上回溯两级找到所需的目录
parent_dir = os.path.dirname(os.path.dirname(current_dir))
# 将找到的目录添加到sys.path
sys.path.append(parent_dir)
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.drop_path import DropPath
import scipy.stats as stats

# diff loss
from diffloss import DiffLoss

from tqdm import tqdm
import numpy as np
import math

# import debugpy; debugpy.connect(("localhost", 6000))


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02

    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = "t2v"

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048

    vae_embed_dim: int = 2048
    t_downsample_size: int = 4
    num_frames: int = 17
    save_train_video_latent: bool = False

    diffloss_d: int = 3
    diffloss_w: int = 1024
    num_sampling_steps: int = "100"
    diffusion_batch_mul: int = 4

    grad_checkpointing: bool = False
    mask_ratio_min: float = 0.7


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
        )
        self.register_buffer(
            "uncond_embedding",
            nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5),
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = (
            config.n_kv_head if config.n_kv_head is not None else config.n_head
        )
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        xq = apply_rotary_emb(xq, freqs_cis)
        xk = apply_rotary_emb(xk, freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        if self.kv_cache is not None:
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv
        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq,
            keys,
            values,
            attn_mask=mask,
            is_causal=(
                True if mask is None else False
            ),  # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: int,
        mask: Optional[torch.Tensor] = None,
    ):
        h = x + self.drop_path(
            self.attention(self.attention_norm(x), freqs_cis, start_pos, mask)
        )
        out = h + self.drop_path(self.feed_forward(self.ffn_norm(h)))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.num_frames = config.num_frames
        self.t_downsample_size = config.t_downsample_size
        # 2d rotary pos embedding
        grid_size = int(self.block_size**0.5)
        assert grid_size * grid_size == self.block_size

        # print(f'{self.model_type=}!')
        if self.model_type == "c2i":
            self.cls_embedding = LabelEmbedder(
                config.num_classes, config.dim, config.class_dropout_prob
            )
        elif self.model_type == "t2i":
            self.cls_embedding = CaptionEmbedder(
                config.caption_dim, config.dim, config.class_dropout_prob
            )
            self.freqs_cis = precompute_freqs_cis_2d(
                grid_size,
                self.config.dim // self.config.n_head,
                self.config.rope_base,
                self.cls_token_num,
            )
        elif self.model_type == "t2v":
            self.cls_embedding = CaptionEmbedder(
                config.caption_dim, config.dim, config.class_dropout_prob
            )  # drop 0.1
            self.vae_latent_adapter = MLP(
                config.vae_embed_dim, config.dim, config.dim
            )  # video latent adapter
            self.freqs_cis = precompute_freqs_cis_3d_video(
                grid_size,
                (config.num_frames - 1) // config.t_downsample_size + 1,
                self.config.dim // self.config.n_head,
                self.config.rope_base,
                self.cls_token_num,
            )
            # self.vae_latent_adapter2 = MLP(
            #     config.dim, config.dim, config.vae_embed_dim
            # )  # video latent adapter
            # self.shuffle_video_tokens = True
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.vae_embed_dim))
            self.vae_embed_dim = config.vae_embed_dim
            self.vae_t = (config.num_frames - 1) // config.t_downsample_size + 1
            self.grid_size = grid_size
            self.seq_len = int(grid_size**2 * self.vae_t)  # videovae token nums
            self.llm_dim = config.dim
            # diff
            self.diffloss_d = config.diffloss_d
            self.diffloss_w = config.diffloss_w
            self.num_sampling_steps = config.num_sampling_steps
            self.diffusion_batch_mul = config.diffusion_batch_mul

            # --------------------------------------------------------------------------
            # Diffusion Loss
            # import ipdb; ipdb.set_trace()
            self.diffloss = DiffLoss(
                target_channels=config.vae_embed_dim,  # vae
                z_channels=config.dim,  # model width
                width=self.diffloss_w,
                depth=self.diffloss_d,
                num_sampling_steps=self.num_sampling_steps,
                grad_checkpointing=config.grad_checkpointing,
            )
            self.diffusion_batch_mul = self.diffusion_batch_mul

            #  variant masking ratio, a left-half truncated Gaussian centered at 100% masking ratio with std 0.25, left is mask_ratio_min and the right is 1.0
            self.mask_ratio_generator = stats.truncnorm(
                (config.mask_ratio_min - 1.0) / 0.25,
                (1.0 - 1.0) / 0.25,
                loc=1.0,
                scale=0.25,
            )

        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)  # drop 0.1

        # transformer blocks
        dpr = [
            x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)
        ]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

        # diff

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(
                max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype
            )

        causal_mask = torch.tril(
            torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)
        )
        self.causal_mask = causal_mask.unsqueeze(0).repeat(
            self.max_batch_size, 1, 1
        )  # (bs, seq, seq)
        grid_size = int(self.config.block_size**0.5)
        assert grid_size * grid_size == self.block_size

        self.freqs_cis = precompute_freqs_cis_3d_video(
            grid_size,
            (self.num_frames - 1) // self.t_downsample_size + 1,
            self.config.dim // self.config.n_head,
            self.config.rope_base,
            self.cls_token_num,
        )
        # self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_shuffle

    def forward_loss(self, z, target, mask):

        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz * seq_len).repeat(self.diffusion_batch_mul)

        # import ipdb; ipdb.set_trace()
        # return torch.mean(z)  # print(target.shape)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def forward_decoder(
        self,
        video_latent,  # torch.Size([2, 256, 2048])
        input_pos: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,  # torch.Size([2, 16*16*1])
        cond_embed: Optional[torch.Tensor] = None,  # torch.Size([2, 120, 768])
        cfg: Optional[float] = 1.0,
    ):

        self.freqs_cis = self.freqs_cis.to(
            video_latent.device
        )  # torch.Size([376, 32, 2])
        # concat text embedding and video latent embeddings ( from Causual Video VAE)

        token_embeddings = self.vae_latent_adapter(
            video_latent
        )  # projection        torch.Size([2, 256, 2048])--->torch.Size([2, 256, 768])

        token_embeddings = torch.cat(
            (cond_embed, token_embeddings), dim=1
        )  # torch.Size([2, 120+256, 768])
        h = self.tok_dropout(token_embeddings)  # torch.Size([2, 1399, 768])

        # if self.training:
        #     freqs_cis = self.freqs_cis[: token_embeddings.shape[1]] # torch.Size([376, 32, 2])
        # else:
        #     # freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        freqs_cis = self.freqs_cis[: token_embeddings.shape[1]].to(h.dtype)

        if attn_mask is None:
            assert attn_mask is None, "attn_mask is None"
            # causal_mask = torch.tril(
            #         torch.ones(token_embeddings.shape[1], token_embeddings.shape[1], dtype=torch.bool)
            #     )
            # attn_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(
            #     token_embeddings.shape[0], 1, 1, 1
            # ).to(h.device)  # (bs, 1, seq, seq)
            # eye_matrix = torch.eye(causal_mask.size(1), causal_mask.size(1), device=device)
            # model.causal_mask[:] = model.causal_mask * (1 - eye_matrix) + eye_matrix

        # import ipdb; ipdb.set_trace()
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, attn_mask)

        # output layers
        h = self.norm(h)  # torch.Size([2, 376, 768])

        return h[:, self.cls_token_num - 1 :]  # torch.Size([2, 257, 768])
        # t2v
        if self.training:
            if video_latent is not None and cond_embed is not None:
                pre_video_latents = h[:, self.cls_token_num - 1 :].contiguous()
                # 有一个问题，就是到底是递归预测，还是双向可见预测
                loss = self.forward_loss(
                    z=pre_video_latents, target=targets_video, mask=visual_token_mask
                )
            else:
                raise ValueError("Either video_latent or idx should be None")

            if save_train_video_latent:
                return h, loss, pre_video_latents
            return h, loss  # , h[:, self.cls_token_num - 1:]
        else:
            return h, None  # , h[:, self.cls_token_num - 1:]

    def forward(
        self,
        idx: Optional[torch.Tensor] = None,
        cond_idx: Optional[torch.Tensor] = None,  # cond_idx_or_embed
        input_pos: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        cond_embed: Optional[torch.Tensor] = None,
        video_latent: Optional[torch.Tensor] = None,  # (N, L-1, D)
        targets_video: Optional[torch.Tensor] = None,  # (N, L, D)
        mask_ratio=1,
        save_train_video_latent=False,
    ):

        mask_ratio = self.mask_ratio_generator.rvs(1)[0]
        if targets_video is not None and video_latent is not None:
            assert (
                video_latent.shape[1] == targets_video.shape[1]
            ), "video_latent and targets_video shape mismatch"

            if self.training:
                video_latent_masked, visual_token_mask, ids_restore, ids_shuffle = (
                    self.random_masking(video_latent, mask_ratio)
                )
                # masked tokens are replaced with the mask_token
                noise_video_latent = self.mask_token.repeat(
                    video_latent.shape[0],
                    video_latent.shape[1] - video_latent_masked.shape[1],
                    1,
                ).to(
                    video_latent.device
                )  # (N, token_num, in_channels)

                input_videos_latent = torch.cat(
                    [video_latent_masked, noise_video_latent], dim=1
                )  # (N, L, D)

                # restore the shuffled video tokens
                input_videos_latent = torch.gather(
                    input_videos_latent,
                    1,
                    ids_restore.unsqueeze(-1).expand(-1, -1, targets_video.shape[-1]),
                )[:, :-1]
                # else:
                # input_videos_latent = video_latent[:, :-1]
                # visual_token_mask = torch.ones(video_latent.shape[:2], device=video_latent.device)

                self.freqs_cis = self.freqs_cis.to(input_videos_latent.device)
                # concat text embedding and video latent embeddings ( from Causual Video VAE)
                cond_embeddings = self.cls_embedding(cond_embed, train=self.training)[
                    :, : self.cls_token_num
                ]  # projection

                token_embeddings = self.vae_latent_adapter(
                    input_videos_latent
                )  # projection

                token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
                h = self.tok_dropout(token_embeddings)  # torch.Size([2, 1399, 768])

        else:
            if cond_embed is not None and video_latent is None:  # prefill in inference
                # token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                token_embeddings = self.cls_embedding(cond_embed, train=self.training)[
                    :, : self.cls_token_num
                ]  # projection

            elif (
                cond_embed is None and video_latent is not None
            ):  # decode_n_tokens(kv cache) in inference
                token_embeddings = self.vae_latent_adapter(video_latent)  # projection
                # token_embeddings = self.tok_embeddings(idx)

            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis

        if self.training:
            freqs_cis = self.freqs_cis[: token_embeddings.shape[1]]
        else:
            # freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
            freqs_cis = self.freqs_cis[input_pos]

        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)

        # output layers
        h = self.norm(h)

        # t2v
        if self.training:
            if video_latent is not None and cond_embed is not None:
                pre_video_latents = h[:, self.cls_token_num - 1 :].contiguous()
                # return h, torch.mean(pre_video_latents)  # , h[:, self.cls_token_num - 1:]   MAR: torch.Size([16384, 1024])  llamagen: torch.Size([10240, 768])  torch.Size([65536, 8])
                # 有一个问题，就是到底是递归预测，还是双向可见预测

                loss = self.forward_loss(
                    z=pre_video_latents, target=targets_video, mask=visual_token_mask
                )
            else:
                raise ValueError("Either video_latent or idx should be None")

            if save_train_video_latent:
                return h, loss, pre_video_latents
            return h, loss  # , h[:, self.cls_token_num - 1:]
        else:
            return h, None  # , h[:, self.cls_token_num - 1:]

    def mask_by_order(self, mask_len, order, bsz, seq_len):
        masking = torch.zeros(bsz, seq_len).cuda()
        masking = torch.scatter(
            masking,
            dim=-1,
            index=order[:, : mask_len.long()],
            src=torch.ones(bsz, seq_len).cuda(),
        ).bool()
        return masking

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.seq_len)))
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def sample_tokens(
        self,
        bsz,
        num_iter=64,
        cfg=1.0,
        cfg_schedule="linear",
        cond_embed=None,
        temperature=1.0,
        attn_mask=None,
        progress=False,
    ):

        assert cond_embed is not None, "cond_embed should be provided"
        # init and sample generation orders
        seq_len = int(self.grid_size**2 * self.vae_t)  # 16*16*1
        mask = torch.ones(bsz, seq_len).cuda()  # (2, 256)
        tokens = (
            self.mask_token.repeat(
                cond_embed.shape[0],
                seq_len,
                1,
            )
            .to(cond_embed.device)
            .to(cond_embed.dtype)
        )  # (N, token_num, in_channels)  torch.Size([2, 256, 2048])

        orders = self.sample_orders(bsz)  # torch.Size([2, 256])
        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        # generate latents
        # import ipdb; ipdb.set_trace()
        for step in tqdm(indices):
            cur_tokens = tokens.clone()
            # class embedding and CFG

            cond_embeddings = self.cls_embedding(cond_embed, train=self.training)[
                :, : self.cls_token_num
            ]  # projection       torch.Size([2, 120, 2048])-->torch.Size([2, 120, 768])

            if not cfg == 1.0:
                # raise ValueError("cfg is not 1.0")
                tokens = torch.cat([tokens, tokens], dim=0)
                # class_embedding = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                cond_embeddings = torch.cat(
                    [cond_embeddings, cond_embeddings], dim=0
                )  # 对吗？
                mask = torch.cat([mask, mask], dim=0)

            # mae encoder
            # x = self.forward_mae_encoder(tokens, mask, cond_embeddings)
            # mae decoder
            # z = self.forward_mae_decoder(x, mask)
            z = self.forward_decoder(
                video_latent=tokens[:, :-1],
                mask=mask,
                attn_mask=attn_mask,
                cond_embed=cond_embeddings,
            )

            # mask ratio for the next round, following MaskGIT and MAGE.
            mask_ratio = np.cos(math.pi / 2.0 * (step + 1) / num_iter)
            mask_len = torch.Tensor([np.floor(self.seq_len * mask_ratio)]).cuda()

            # masks out at least one for the next iteration
            mask_len = torch.maximum(
                torch.Tensor([1]).cuda(),
                torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len),
            )

            # get masking for next iteration and locations to be predicted in this iteration
            mask_next = self.mask_by_order(mask_len[0], orders, bsz, seq_len)
            if step >= num_iter - 1:
                mask_to_pred = mask[:bsz].bool()
            else:
                mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if not cfg == 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)

            # sample token latents for this step
            z = z[mask_to_pred.nonzero(as_tuple=True)]  # torch.Size([pred_nums, 768])
            # cfg schedule follow Muse
            if cfg_schedule == "linear":
                cfg_iter = 1 + (cfg - 1) * (self.seq_len - mask_len[0]) / self.seq_len
            elif cfg_schedule == "constant":
                cfg_iter = cfg
            else:
                raise NotImplementedError

            # import ipdb; ipdb.set_trace()
            sampled_token_latent = self.diffloss.sample(
                z, temperature, cfg_iter
            )  # torch.Size([2, 2048])
            if not cfg == 1.0:
                sampled_token_latent, _ = sampled_token_latent.chunk(
                    2, dim=0
                )  # Remove null class samples
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        # tokens = self.unpatchify(tokens)
        # import ipdb; ipdb.set_trace()
        return tokens

    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)


##################################################################################
#                      shuffle and restore video                                 #
##################################################################################
def shuffle_and_restore(N, L, dim, transformer):
    import torch

    # 假设 x 是你的输入张量，形状为 (N, L, dim)
    x = torch.randn(N, L, dim)  # 示例输入

    # Step 1: 使用 torch.argsort 打乱顺序
    # 生成随机顺序的索引
    random_indices = torch.argsort(torch.randn(N, L), dim=1)  # (N, L)

    # 使用这些索引打乱输入张量的顺序
    x_shuffled = torch.gather(x, 1, random_indices.unsqueeze(-1).expand(-1, -1, dim))

    # Step 2: 将打乱后的张量输入 Transformer
    # 这里假设 transformer 是一个你定义的 Transformer 模型
    output_shuffled = transformer(x_shuffled)

    # Step 3: 还原原始顺序
    # 获取逆排序索引
    _, inverse_indices = random_indices.sort(dim=1)

    # 使用这些索引还原经过 Transformer 处理后的张量的顺序
    output_original = torch.gather(
        output_shuffled, 1, inverse_indices.unsqueeze(-1).expand(-1, -1, dim)
    )

    # 现在 output_original 的顺序与 x 的原始顺序相同


#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack(
        [freqs_cis.real, freqs_cis.imag], dim=-1
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_2d(
    grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120
):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    cache = cache_grid.flatten(0, 1)
    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), cache]
    )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def precompute_freqs_cis_3d_video(
    grid_size: int, vae_t, n_elem: int, base: int = 10000, cls_token_num=120
):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (
        base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim)
    )
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs)  # (grid_size, head_dim // 2)
    freqs_grid = torch.concat(
        [
            freqs[:, None, :].expand(-1, grid_size, -1),
            freqs[None, :, :].expand(grid_size, -1, -1),
        ],
        dim=-1,
    )  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack(
        [torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1
    )  # (grid_size, grid_size, head_dim // 2, 2)
    # import ipdb; ipdb.set_trace()
    cache = cache_grid.flatten(0, 1)  # (grid_size*grid_size, head_dim // 2, 2)

    # 使用 repeat 方法来重复 t 次
    repeated_cache = cache.unsqueeze(0).repeat(
        (vae_t, 1, 1, 1)
    )  # 形状变为 (t, g * g, c, 2)
    # 然后使用 view 或 reshape 来展平前两维
    flattened_cache = repeated_cache.flatten(0, 1)  # 形状变为 (t * g * g, c, 2)

    cond_cache = torch.cat(
        [torch.zeros(cls_token_num, n_elem // 2, 2), flattened_cache]
    )  # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2)
    xshaped = x.float().reshape(
        *x.shape[:-1], -1, 2
    )  # (bs, seq_len, n_head, head_dim//2, 2)
    freqs_cis = freqs_cis.view(
        1, xshaped.size(1), 1, xshaped.size(3), 2
    )  # (1, seq_len, 1, head_dim//2, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        dim=-1,
    )
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs))  # 6.6B


def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs))  # 3.1B


def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs))  # 1.2B


### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs))  # 3.9B


def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs))  # 1.4B


def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs))  # 775M


def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs))  # 343M


def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs))  # 111M


GPT_models = {
    "GPT-B": GPT_B,
    "GPT-L": GPT_L,
    "GPT-XL": GPT_XL,
    "GPT-XXL": GPT_XXL,
    "GPT-XXXL": GPT_XXXL,
    "GPT-1B": GPT_1B,
    "GPT-3B": GPT_3B,
    "GPT-7B": GPT_7B,
}


def video_latent_permute(video_latent):
    # (bs, dim ,t, h, w)
    # 使用 permute 调整维度顺序
    video_latent = video_latent.permute(0, 2, 3, 4, 1)  # (bs, t, h, w, dim)
    # 使用 reshape 将 t, h, w 维度展平
    video_latent = video_latent.reshape(
        video_latent.shape[0], -1, video_latent.shape[-1]
    )  # (bs, t*h*w, dim)
    return video_latent


from einops import rearrange, repeat

import sys

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
sys.path.append(os.path.join(current_directory, ".."))
sys.path.append(os.path.join("./", "CausalVideoVAE"))
from dataset.transform import (
    ToTensorVideo,
    TemporalRandomCrop,
    RandomHorizontalFlipVideo,
    CenterCropResizeVideo,
    LongSideResizeVideo,
    SpatialStrideCropVideo,
)

# from CausalVideoVAE.causalvideovae.model import ae_norm, ae_denorm
from torchvision.transforms import Lambda
import argparse
import torchvision.transforms as transforms
from language.t5 import T5Embedder
from dataset.t2v import T2V_dataset
from torch.utils.data import Dataset, DataLoader
from tokenizer.tokenizer_image.vae_model import VAE_models

import inspect
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # dataset & dataloader
    # parser.add_argument("--dataset", type=str, required=True)
    # parser.add_argument("--data", type=str, required='')
    parser.add_argument(
        "--video_meta_info_file",
        type=str,
        default="/storage/zhubin/liuyihang/add_aes/output/sucai_aes_1000.json",
    )
    # parser.add_argument("--sample_rate", type=int, default=1)
    # parser.add_argument("--cache_dir", type=str, required='')
    parser.add_argument(
        "--t5-model-path", type=str, default="./pretrained_models/t5-ckpt"
    )
    parser.add_argument("--t5-model-type", type=str, default="flan-t5-xl")
    parser.add_argument("--max_height", type=int, default=256)
    parser.add_argument("--max_width", type=int, default=256)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--start_frame_ind", type=int, default=25)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--data_root", type=str, default="/storage/dataset")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--t-downsample-size", type=int, choices=[4, 8], default=4)

    parser.add_argument(
        "--vae-model", type=str, choices=list(VAE_models.keys()), default="VAE-16"
    )
    parser.add_argument(
        "--vae-ckpt",
        type=str,
        default="/storage/clh/models/488dim8",
        help="ckpt path for vq model",
    )
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--t5-path", type=str, required=True)  # t5 model path

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    
    args = parser.parse_args()

    resize = [
        CenterCropResizeVideo((args.max_height, args.max_width)),
    ]
    # norm_fun = ae_norm[args.ae]
    norm_fun = Lambda(lambda x: 2.0 * x - 1.0)
    transform = transforms.Compose(
        [
            ToTensorVideo(),
            *resize,
            # RandomHorizontalFlipVideo(p=0.5),  # in case their caption have position decription
            norm_fun,
        ]
    )
    from dataset.augmentation import center_crop_arr

    assert os.path.exists(args.t5_model_path)
    device = "cuda"
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]

    t5_xxl = T5Embedder(
        device=device,
        local_cache=True,
        cache_dir=args.t5_model_path,
        dir_or_name=args.t5_model_type,
        torch_dtype=precision,
    )

    dataset = T2V_dataset(args, transform, t5_xxl)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=24)
    # return dict(video=video, input_ids=input_ids, cond_mask=cond_mask)
    CausalVAEModel = VAE_models[args.vae_model]
    vae = CausalVAEModel.from_pretrained(args.vae_ckpt)

    # import ipdb; ipdb.set_trace()
    if args.enable_tiling:
        vae.enable_tiling()
        vae.tile_overlap_factor = args.tile_overlap_factor
    vae = vae.to(device)  # .to(data_type)
    vae.eval()

    # import ipdb; ipdb.set_trace()
    latent_size = args.image_size // args.downsample_size

    model = GPT_models["GPT-B"](
        vocab_size=16384,
        block_size=latent_size**2,
        num_classes=1000,
        cls_token_num=120,
        resid_dropout_p=0.1,
        ffn_dropout_p=0.1,
        token_dropout_p=0.1,
        model_type="t2v",
        vae_embed_dim=vae.config.embed_dim,
        num_frames=args.num_frames,
        t_downsample_size=args.t_downsample_size,
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision =='fp16'))
    from utils.logger import create_logger
    # logger = create_logger('tmp_experiment_dir')
    import logging
    logger =  logging.getLogger('tmp_experiment_dir')
    logger.addHandler(logging.NullHandler())
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)
    train_steps = 0
    running_loss = 0
    log_steps = 0
    import time; start_time = time.time()

    for sample in dataloader:
        x, y, attn_mask, valid = (
            sample["video_data"]["video"],
            sample["video_data"]["t5_feat_padding"],
            sample["video_data"]["attn_mask"],
            sample["video_data"]["valid"],
        )
        #     print(x.shape)
        # for i in range(1000):
        #     x = torch.randn(10, 3, 1, 256, 256).to(device)
        #     y = torch.randn(10, 1, 120, 2048).to(device)
        #     attn_mask = torch.randn(10, 1, 376, 376).to(device)
        #     valid = torch.randn(10).to(device)

        x = x.to(device)
        y = y.to(device)
        attn_mask = attn_mask.to(device)
        valid = valid.to(device)

        c_indices = y.reshape(
            y.shape[0], y.shape[-2], y.shape[-1]
        )  # torch.Size([2, 120, 2048])
        """
        x: torch.Size([2, 3, 1, 256, 256])
        y: torch.Size([2, 1, 120, 2048])
        attn_mask: torch.Size([2, 1, 376, 376])
        valid: torch.Size([2])
        """

        # import ipdb; ipdb.set_trace()
        with torch.no_grad():
            z = vae.encode(x).sample()
        # torch.Size([2, 2048, 5, 16, 16])
        video_latent = z.flatten(2).transpose(
            1, 2
        )  # (b, c, t, h, w)  ->  (b, t*h*w, c)
        # video_latent = torch.randn(10, 256, 2048).to(device)

        cond_embed = c_indices

        attn_mask = attn_mask.reshape(
            attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1]
        )  # (bs, n_head, seq_len, seq_len)

        with torch.cuda.amp.autocast(dtype=precision):
            # tokens = model.sample_tokens(
            #     cond_embed.shape[0], num_iter=64, cfg=1.0, cfg_schedule="linear",
            #     cond_embed=cond_embed, temperature=1.0, progress=False
            # )
            # continue
            _, loss = model(
                idx=None,
                cond_idx=None,
                targets=None,
                # cond_idx=c_indices,
                # idx=z_indices[:, :-1],
                # targets=z_indices,
                mask=attn_mask[:, :, :-1, :-1],  # torch.Size([2, 1, 1400, 1400])
                valid=valid,  # torch.Size([2])
                video_latent=video_latent,  # [:, :-1],  # torch.Size([2, 1280, 2048])
                targets_video=video_latent,
                cond_embed=cond_embed,  # torch.Size([2, 120, 2048])
            )

            print(loss)

            scaler.scale(loss).backward()
            if args.max_grad_norm != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            import time; import torch.distributed as dist
            if train_steps % 1 == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                # avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

    """ 

    pkill -9 -f train_t2v.py

    source  /storage/miniconda3/etc/profile.d/conda.sh 
    conda activate motionctrl
    export CUDA_VISIBLE_DEVICES=5
    cd /storage/zhubin/LlamaGen 
    python autoregressive/models/gpt_video_mask_diff.py --t5-path  /storage/zhubin/LlamaGen/dataset/storage_datasets_npy \
    --video_meta_info_file  /storage/zhubin/video_statistics_data/task1.5/Final_format_dataset_data_v2/step1.5_istock_final_815070_random_3000.json \
    --num_frames 1   \
    --downsample-size 8 \
    --precision none 
    
    print(z.shape)

    """
