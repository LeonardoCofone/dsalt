"""DSALT sparse attention: adaptive local window plus landmarks chosen by content.

Builds the differentiable selectors (a soft window edge trains ``win_gate``, a soft
landmark weight trains the per-head ``alpha``) and dispatches to the Triton training
and inference kernels on CUDA, or to a masked-SDPA fallback elsewhere. The selector
math is defined once in ``selectors`` and ``landmark_tokens_ker``.
"""

import math
import warnings
import torch
import torch.nn as nn

from ..kernels.window_utils import (
    build_local_window_mask,
    build_local_window_mask_packed,
    build_rope_cache,
    apply_rotary_emb,
)
from ..kernels.sparse_attn import (
    sparse_attention_forward,
    sparse_attention_forward_packed,
)
from ..kernels.landmark_tokens_ker import hybrid_scores_per_head
# The selectors do not depend on Triton, so they import on CPU too.
from ..kernels.selectors import _compute_landmark_indices

try:
    from ..kernels.dsalt_triton_attn import dsalt_triton_attention
    _TRITON_OK = True
except Exception:
    _TRITON_OK = False

try:
    from ..kernels.dsalt_triton_train import dsalt_triton_train_attention
    _TRITON_TRAIN_OK = True
except Exception:
    _TRITON_TRAIN_OK = False


def _dynamo_opaque(fn):
    """Make ``fn`` opaque to torch.compile (graph break, eager execution).

    Autograd is unaffected; only tracing is skipped. Identity on torch builds without
    ``_dynamo``.
    """
    disable = getattr(getattr(torch, "_dynamo", None), "disable", None)
    return disable(fn) if disable is not None else fn


@torch.no_grad()
def _hybrid_scores(
    x:     torch.Tensor,
    W_V:   torch.Tensor,
    alpha: torch.Tensor,
    dh:    int,
) -> torch.Tensor:
    # The score formula is defined once, in kernels.landmark_tokens_ker.
    n_heads      = alpha.shape[0]
    scores, _, _ = hybrid_scores_per_head(x, W_V, alpha, n_heads, dh)
    return scores

@torch.no_grad()
def _build_mask_batched(
    x:        torch.Tensor,
    w_sizes:  torch.Tensor,
    v_weight: torch.Tensor,
    alpha:    torch.Tensor,
    k_lmk:   int,
    T:        int,
    device:   torch.device,
    dh:       int | None = None,
) -> torch.Tensor:
    n_heads = alpha.shape[0]
    # Under GQA v_weight has only n_kv_heads head rows, so the per-head dim is passed
    # explicitly.
    if dh is None:
        dh = v_weight.shape[0] // n_heads

    window_mask   = build_local_window_mask(T, w_sizes, device)
    scores        = _hybrid_scores(x, v_weight, alpha, dh)
    in_window     = window_mask.any(dim=0)
    scores_fil    = scores.masked_fill(in_window.unsqueeze(1), float("-inf"))
    k_act         = min(k_lmk, int((scores_fil > float("-inf")).sum(dim=0).min().item()))
    landmark_mask = window_mask.unsqueeze(0).expand(n_heads, -1, -1).clone()

    if k_act > 0:
        _, idx  = torch.topk(scores_fil, k_act, dim=0, sorted=False)
        idx_exp = idx.T.unsqueeze(1).expand(n_heads, T, k_act)
        landmark_mask.scatter_(2, idx_exp, True)

    return landmark_mask


@torch.no_grad()
def _build_mask_packed(
    x:          torch.Tensor,
    w_sizes:    torch.Tensor,
    v_weight:   torch.Tensor,
    alpha:      torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_len:  int,
    k_lmk:     int,
    dh:         int,
    device:     torch.device,
) -> torch.Tensor:
    mask     = build_local_window_mask_packed(cu_seqlens, w_sizes, total_len, device)
    scores   = _hybrid_scores(x, v_weight, alpha, dh)

    num_seqs = cu_seqlens.shape[0] - 1
    lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
    starts   = cu_seqlens[:-1].to(device)
    seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
    seq_off  = torch.arange(total_len, device=device) - starts[seq_ids]
    max_len  = int(lens.max())

    in_win_global = (seq_off < w_sizes.long()).unsqueeze(1)
    s_masked      = scores.masked_fill(in_win_global, float("-inf"))

    score_pad = torch.full((num_seqs, max_len, alpha.shape[0]), float("-inf"), device=device)
    score_pad[seq_ids, seq_off] = s_masked

    k_eff     = min(k_lmk, max_len)
    vals, top = torch.topk(score_pad, k_eff, dim=1, sorted=False)
    valid     = vals > float("-inf")

    abs_lmk = (starts[:, None, None] + top).clamp(max=total_len - 1)

    n_heads = alpha.shape[0]
    rows_per_seq = lens.long()
    row_starts   = cu_seqlens[:-1].to(device).long()

    for b in range(num_seqs):
        if not valid[b].any():
            continue
        s   = int(cu_seqlens[b])
        e   = int(cu_seqlens[b + 1])
        idx = abs_lmk[b][valid[b]].reshape(-1).unique()

        row_idx = torch.arange(s, e, device=device).unsqueeze(1).expand(-1, idx.shape[0])
        col_idx = idx.unsqueeze(0).expand(e - s, -1)
        mask[row_idx, col_idx] = True

    return mask


class DSALTAttention(nn.Module):
    """DSALT sparse attention: adaptive local window plus landmark tokens.

    Query i attends to ``A(i) = W(i) ∪ L(i)``: a causal local window of
    ``w(i) = n_min + σ(f(x_i))·(n_max-n_min)`` tokens, predicted by ``win_gate`` from
    the block input, and the landmarks that precede the window, selected by the hybrid
    score with per-head balance ``alpha_w``.

    Landmarks are selected once per sequence and head: the score is standardized over
    the whole sequence and the top-k is taken over all positions. A query never attends
    to a later token, but which earlier tokens are landmarks depends on later tokens, so
    the selection is not causal in the strict sense.

    ``win_gate`` and ``alpha_w`` are trained. The selection itself is hard; the gradient
    reaches the two predictors through soft weights:
      * soft window edge: the last ``win_edge`` keys at the window border get weight
        ``σ((w̃(i)-d)/τ)``, continuous in ``w̃`` (trains ``win_gate``);
      * soft landmark weight: each selected landmark's logit gets ``log σ(s_j(α)/τ)``,
        continuous in ``α`` (trains ``alpha_w``).
    The union is the elementwise max of the two log-biases; ``q·k`` is never biased by
    the score.

    Training uses a differentiable path (Triton kernel on CUDA, dense SDPA otherwise);
    inference uses ``dsalt_triton_attention`` (packed) or a dense hard mask (batched).
    In eval mode ``_last_P`` stores the dense attention matrix of the first sequence,
    used by the trainer's metrics.
    """

    def __init__(
        self,
        d_model:     int,
        n_heads:     int,
        n_min:       int,
        n_max:       int,
        k_lmk:       int,
        max_seq_len: int,
        dropout:     float = 0.0,
        yarn_scale:  float = 1.0,
        layer_idx:   int   = 0,
        n_kv_heads:  int | None = None,
        qkv_bias:    bool   = False,
        rope_base:   float  = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model     = d_model
        self.n_heads     = n_heads
        self.head_dim    = d_model // n_heads
        # Grouped-query attention: n_kv_heads key/value heads shared by the n_heads
        # query heads (None: full MHA). The kv heads are repeated to n_heads right after
        # the projections, so the kernels, the selectors and the SDPA fallback all see
        # an n_heads layout; the unexpanded count only sizes k_proj / v_proj, so that a
        # GQA backbone such as Qwen2.5 loads unchanged.
        self.n_kv_heads  = n_kv_heads if n_kv_heads is not None else n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_rep       = n_heads // self.n_kv_heads
        self.kv_dim      = self.n_kv_heads * self.head_dim
        self.n_min       = n_min
        self.n_max       = n_max
        self.k_lmk       = k_lmk
        self.max_seq_len = max_seq_len
        self.dropout     = dropout
        self.yarn_scale  = yarn_scale
        self.rope_base   = rope_base
        self.layer_idx   = layer_idx

        # q is full width; k and v are kv_dim wide. Qwen2.5 has q/k/v biases.
        self.q_proj   = nn.Linear(d_model, d_model,      bias=qkv_bias)
        self.k_proj   = nn.Linear(d_model, self.kv_dim,  bias=qkv_bias)
        self.v_proj   = nn.Linear(d_model, self.kv_dim,  bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model,      bias=False)

        # Adaptive window predictor f: R^d -> R, w(i) = n_min + σ(f(x_i))·(n_max-n_min),
        # computed from the block input. Trained through the soft window edge (see
        # _soft_window_logbias).
        self.win_gate = nn.Linear(d_model, 1, bias=True)

        # Per-head balance α = σ(alpha_w), initialized at 0.6. The top-k selection is
        # hard; α is trained through the σ(s_j(α)/τ) weight of the selected landmarks
        # (see _soft_landmark_logbias).
        self.alpha_w = nn.Parameter(torch.full((n_heads,), math.log(0.6 / 0.4)))

        # tau_win and tau_lmk set the sharpness of the soft window edge and of the
        # landmark weight. win_edge is the width, in tokens, of the soft band at the
        # window border; inside it the bias is exactly 0.
        self.tau_win  = 1.0
        self.tau_lmk  = 2.0 
        self.win_edge = 4

        self._last_P: torch.Tensor | None = None

        cos, sin = build_rope_cache(max_seq_len, self.head_dim, torch.device("cpu"), base=rope_base, scale=yarn_scale)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_w)

    def _apply_out_proj(self, out: torch.Tensor) -> torch.Tensor:
        """Output projection, with ``out`` cast to the weight dtype.

        Some paths leave ``out`` in fp32 while the weight is bf16, and the metrics probe
        runs without autocast, where addmm requires matching dtypes.
        """
        return self.out_proj(out.to(self.out_proj.weight.dtype))

    def _repeat_kv(self, t: torch.Tensor) -> torch.Tensor:
        """Repeat grouped key/value heads to ``n_heads`` (no-op without GQA).

        ``t`` is ``[..., n_kv_heads, head_dim]`` (packed) or ``[B, n_kv_heads, T, head_dim]``
        (batched). The expansion is materialized so the kernels and the SDPA fallback see a
        plain ``n_heads`` layout.
        """
        if self.n_rep == 1:
            return t
        if t.dim() == 3:                        # [T, n_kv_heads, D] (packed)
            T, _, D = t.shape
            return t.unsqueeze(2).expand(T, self.n_kv_heads, self.n_rep, D).reshape(T, self.n_heads, D)
        # [B, n_kv_heads, T, D] (batched)
        B, _, T, D = t.shape
        return t.unsqueeze(2).expand(B, self.n_kv_heads, self.n_rep, T, D).reshape(B, self.n_heads, T, D)

    def _window_continuous(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous window size w̃(i) = n_min + σ(f(x_i))·(n_max - n_min), differentiable in
        ``win_gate``. Returns ``[T]`` floats in ``[n_min, n_max]``.
        """
        win_logits = self.win_gate(x).squeeze(-1)                                  # [T]
        return self.n_min + torch.sigmoid(win_logits) * (self.n_max - self.n_min)

    @torch.no_grad()
    def _window_sizes(self, x: torch.Tensor, floor: bool) -> torch.Tensor:
        """Discrete window size w(i) for hard masks: floor at inference, round otherwise.
        Returns an integer-valued float tensor ``[T]``, at least 1.
        """
        w_cont = self._window_continuous(x)
        w_disc = w_cont.floor() if floor else w_cont.round()
        return w_disc.clamp(min=1)

    def _soft_window_logbias(self, w_cont: torch.Tensor, T: int, device: torch.device) -> torch.Tensor:
        """Differentiable log-bias of the local window: hard core plus soft edge.

        For query i and key j let ``d = i - j``. With ``w̃ = w_cont[i]`` and ``B = win_edge``:

          * ``d < w̃ - B``: inside the core, bias 0;
          * near ``w̃``:    soft edge, bias log σ((w̃ - d)/τ);
          * far outside:   bias tends to -inf through the σ tail;
          * ``d < 0``:     -inf.

        Returns a ``[T, T]`` additive log-bias in float32.
        """
        idx  = torch.arange(T, device=device)
        d    = (idx.unsqueeze(1) - idx.unsqueeze(0)).float()      # [T,T] = i - j
        w_i  = w_cont.unsqueeze(1)                                # [T,1], broadcast over j
        # σ((w̃ - d)/τ): close to 1 inside the window, to 0 outside, smooth at the
        # border.
        g    = torch.sigmoid((w_i - d) / self.tau_win)           # [T,T] in (0,1)
        logb = torch.log(g.clamp(min=1e-20))
        # Inside the core (d ≤ w̃ - win_edge) the bias is exactly 0, so the gradient on
        # w̃ comes from the border band only. ``+ 0·logb`` keeps autograd connected.
        core = d <= (w_i - self.win_edge)
        logb = torch.where(core, torch.zeros_like(logb), logb)
        # Keys after the query (d < 0) are never attended.
        logb = logb.masked_fill(d < 0, float("-inf"))
        return logb

    def _soft_landmark_logbias(
        self,
        scores:    torch.Tensor,
        lmk_mask:  torch.Tensor,
        T:         int,
    ) -> torch.Tensor:
        """Differentiable log-bias of the landmarks: hard selection plus soft α-weight.

        ``scores`` ``[T, H]`` is the hybrid score s_j(α), continuous in ``α``; ``lmk_mask``
        ``[H, T, T]`` is the hard, detached top-k selection. Selected entries get
        ``log σ(s_j / τ)``, the others ``-inf``. Returns ``[H, T, T]``.
        """
        # The weight depends only on the key j.
        w_lmk = torch.log(torch.sigmoid(scores / self.tau_lmk).clamp(min=1e-20))  # [T,H]
        w_lmk = w_lmk.transpose(0, 1).unsqueeze(1).expand(self.n_heads, T, T)     # [H,T,T] over key j
        neg_inf = torch.full_like(w_lmk, float("-inf"))
        return torch.where(lmk_mask, w_lmk, neg_inf)

    @torch.no_grad()
    def _rope(self, n: int, device: torch.device):
        max_n = self.rope_cos.shape[0]
        if n <= max_n:
            return self.rope_cos[:n].to(device), self.rope_sin[:n].to(device)
        return self.rope_cos.to(device), self.rope_sin.to(device)

    def _aux_zero(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        # There is no auxiliary loss: win_gate and alpha_w are trained through the soft
        # window edge and the soft landmark weight. The (out, aux) signature is kept
        # with a zero.
        return torch.zeros((), device=device, dtype=dtype)

    @torch.no_grad()
    def warmup(self, device: torch.device) -> None:
        was_training = self.training
        self.eval()
        dummy_T = max(self.n_min * 2, 16)
        x_w     = torch.randn(dummy_T, self.d_model, device=device)
        cu_w    = torch.tensor([0, dummy_T], dtype=torch.int32, device=device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = self._packed(x_w, cu_w, dummy_T, device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.train(was_training)

    def forward(
        self,
        x:          torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None          = None,
        rope_cs:    tuple | None         = None,
        cu_list:    list | None          = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cu_seqlens is not None:
            return self._packed(x, cu_seqlens, x.shape[0], x.device, rope_cs, cu_list)
        B, T, _ = x.shape
        return self._batched(x, B, T, x.device)

    def _batched(self, x: torch.Tensor, B: int, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x).view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self._rope(T, device)
        # RoPE on the kv heads, then repeat them to n_heads (no-op without GQA).
        q, k     = apply_rotary_emb(q, k, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0))
        k        = self._repeat_kv(k)
        v        = self._repeat_kv(v)

        x_1b = x[0] if B > 1 else x.squeeze(0)
        scale = 1.0 / math.sqrt(self.head_dim)

        # Differentiable sparse mask. win_logbias [T,T] trains win_gate through the soft
        # window border; lmk_logbias [H,T,T] trains alpha through the weight of the
        # selected landmarks. The union is the elementwise max of the two biases.
        win_logbias, lmk_logbias = self._diff_logbias(x_1b, T, device)             # [T,T],[H,T,T]
        logbias = torch.maximum(win_logbias.unsqueeze(0), lmk_logbias)            # [H,T,T]

        sc  = torch.matmul(q[0], k[0].transpose(-2, -1)).float() * scale          # [H,T,T]
        P   = torch.softmax(sc + logbias, dim=-1)                                  # [H,T,T]
        if self.training and self.dropout > 0:
            P = torch.dropout(P, self.dropout, train=True)
        out = torch.matmul(P.to(v.dtype), v[0])                                    # [H,T,D]
        out = out.unsqueeze(0)                                                     # [1,H,T,D]

        aux = self._aux_zero(x.device, x.dtype)
        self._last_P = P.detach() if not self.training else None

        return self._apply_out_proj(out.transpose(1, 2).contiguous().view(B, T, self.d_model)), aux

    def _diff_logbias(self, x: torch.Tensor, T: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The two differentiable log-biases for [T] tokens.

        Returns ``(win_logbias [T,T], lmk_logbias [H,T,T])``. The landmark selection is
        computed under ``no_grad``; only the weight σ(s/τ) carries the α gradient.
        """
        w_cont = self._window_continuous(x)                                       # [T]
        alpha  = self._alpha()                                                     # [H]
        scores, _, _ = hybrid_scores_per_head(                                     # [T,H]
            x, self.v_proj.weight, alpha, self.n_heads, self.head_dim,
        )

        win_logbias = self._soft_window_logbias(w_cont, T, device)               # [T,T]

        # Hard landmark selection, detached. The window exclusion is per query (a key
        # inside the window of one query can be a landmark for a farther one), so the
        # top-k is taken over the whole sequence and the exclusion is applied per (i,
        # j), as in the Triton kernel (`lmk_pos < win_lo`).
        with torch.no_grad():
            w_disc = w_cont.floor().clamp(min=1)                                  # [T]
            idx    = torch.arange(T, device=device)
            d      = idx.unsqueeze(1) - idx.unsqueeze(0)                          # [T,T] = i - j
            k_act  = min(self.k_lmk, T)
            # landmark keys per head [H,T]
            sel_hk = torch.zeros(self.n_heads, T, dtype=torch.bool, device=device)
            if k_act > 0:
                _, top = torch.topk(scores.detach(), k_act, dim=0, sorted=False)  # [k,H] key indices
                sel_hk.scatter_(1, top.transpose(0, 1), True)                     # [H,T]
            # validity per query (causal and outside the window), the same for every
            # head
            valid_ij = (d >= 0) & (d >= w_disc.unsqueeze(1))                      # [T,T]
            lmk_mask = sel_hk.unsqueeze(1) & valid_ij.unsqueeze(0)               # [H,T,T]

        # -inf where there is no landmark, log σ(s/τ) where there is one
        lmk_logbias = self._soft_landmark_logbias(scores, lmk_mask, T)           # [H,T,T]
        return win_logbias, lmk_logbias

    def _packed_train_triton(self, q, k, v, x, cu_seqlens, total_len, device, cu_list=None):
        """Differentiable sparse training with the Triton kernel (GPU only).

        Inputs to the kernel:
          * ``w_cont`` ``[total_len]``: continuous window (differentiable in ``win_gate``);
          * ``lmk_pos`` ``[H,num_seqs,k]``: hard landmark indices, as in the inference kernel;
          * ``lmk_logw`` ``[H,num_seqs,k]``: log σ(s_j(α)/τ_lmk) at the selected landmarks
            (differentiable in ``alpha``).
        ``cu_list`` is a host copy of cu_seqlens, to avoid device-to-host syncs.
        """
        w_cont, lmk_pos, lmk_logw = self._train_selectors(x, cu_seqlens, total_len, device, cu_list)
        return dsalt_triton_train_attention(
            q, k, v, lmk_pos, lmk_logw, w_cont, cu_seqlens,
            tau_win=self.tau_win, win_edge=float(self.win_edge),
            n_max=self.n_max, cu_list=cu_list,
        )

    @_dynamo_opaque
    def _train_selectors(self, x, cu_seqlens, total_len, device, cu_list=None):
        """Selectors shared by the Triton kernel and the dense fallback.

        Opaque to torch.compile: it branches on Python ints derived from ``cu_list`` that
        change every step with variable-length packing, which would force a re-trace per
        step. Gradients to ``win_gate`` and ``alpha`` are unaffected.

        Returns ``(w_cont [total_len], lmk_pos [H,S,k], lmk_logw [H,S,k])``, consumed by
        both paths so that they select the same landmarks.
        """
        w_cont = self._window_continuous(x)                                       # [total_len]
        alpha  = self._alpha()                                                     # [H]
        with torch.no_grad():
            w_disc = w_cont.floor().clamp(min=1)
            lmk_pos, _, _ = _compute_landmark_indices(
                x.detach(), self.v_proj.weight.detach(), alpha.detach().float(),
                w_disc, cu_seqlens, self.k_lmk, self.n_min, total_len, cu_list,
                head_dim=self.head_dim,
            )                                                                     # [H,num_seqs,k]
        scores, _, _ = hybrid_scores_per_head(                                     # [total_len,H]
            x, self.v_proj.weight, alpha, self.n_heads, self.head_dim,
        )
        starts   = cu_seqlens[:-1].to(device)
        safe_pos = lmk_pos.clamp(min=0)                                            # [H,S,k]
        abs_pos  = starts[None, :, None] + safe_pos                               # [H,S,k]
        head_ix  = torch.arange(self.n_heads, device=device)[:, None, None]
        s_sel    = scores[abs_pos, head_ix]                                        # [H,S,k]
        lmk_logw = torch.log(torch.sigmoid(s_sel / self.tau_lmk).clamp(min=1e-20))
        lmk_logw = torch.where(lmk_pos >= 0, lmk_logw, torch.full_like(lmk_logw, float("-inf")))
        return w_cont, lmk_pos, lmk_logw

    def _packed_train(self, q, k, v, x, cu, lens, total_len, device, cu_seqlens=None):
        """Dense differentiable training attention, the reference for the Triton kernel.

        Builds the same log-bias as the kernel (hard band ``d ≤ w̃`` with soft edge,
        landmarks at ``lmk_pos`` weighted by ``lmk_logw``, union by elementwise max) from
        the selectors of :meth:`_train_selectors`, and runs SDPA. Uniform lengths only.
        """
        H, D = self.n_heads, self.head_dim
        uniform = len(lens) > 0 and all(l == lens[0] for l in lens)
        if cu_seqlens is None:
            cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
        if not uniform:
            raise NotImplementedError(
                "_packed_train dense fallback supports uniform-length packing only "
                "(used for CPU tests / kernel verification)."
            )

        N, L = len(lens), lens[0]
        # Same selectors as the kernel, hence the same landmarks and weights.
        w_cont, lmk_pos, lmk_logw = self._train_selectors(x, cu_seqlens, total_len, device, cu)
        logbias = self._dense_logbias_from_kernel(w_cont, lmk_pos, lmk_logw, N, L, device)  # [N,H,L,L]

        qb = q.view(N, L, H, D).transpose(1, 2)                                    # [N,H,L,D]
        kb = k.view(N, L, H, D).transpose(1, 2)
        vb = v.view(N, L, H, D).transpose(1, 2)
        ob = torch.nn.functional.scaled_dot_product_attention(
            qb, kb, vb, attn_mask=logbias.to(qb.dtype),
            dropout_p=self.dropout if self.dropout > 0 else 0.0,
        )                                                                         # [N,H,L,D]
        return ob.transpose(1, 2).reshape(total_len, H, D)

    def _dense_logbias_from_kernel(self, w_cont, lmk_pos, lmk_logw, N, L, device):
        """Dense ``[N,H,L,L]`` log-bias with the same math as the Triton kernel.

        * window: hard band ``0 ≤ d ≤ w̃`` with soft edge ``log σ((w̃−d)/τ_win)``, 0 in the
          core ``d ≤ w̃−win_edge``;
        * landmarks: ``lmk_logw`` at ``lmk_pos`` where ``lmk_pos ≤ i`` and
          ``lmk_pos < i − w̃``;
        * union: elementwise max.
        """
        H = self.n_heads
        w_seq = w_cont.view(N, L)                                                  # [N,L] per query i
        idx   = torch.arange(L, device=device)
        d     = (idx.unsqueeze(1) - idx.unsqueeze(0)).float()                      # [L,L] = i - j

        # window band (trains win_gate)
        w_i    = w_seq.unsqueeze(-1)                                               # [N,L,1]
        z      = (w_i - d.unsqueeze(0)) / self.tau_win
        win_lb = torch.log(torch.sigmoid(z).clamp(min=1e-20))
        core   = d.unsqueeze(0) <= (w_i - self.win_edge)
        win_lb = torch.where(core, torch.zeros_like(win_lb), win_lb)
        in_band = (d.unsqueeze(0) >= 0) & (d.unsqueeze(0) <= w_i)
        win_lb = win_lb.masked_fill(~in_band, float("-inf"))                       # [N,L,L]

        # landmarks (trains alpha). For query i and selected landmark p the bias is
        # lmk_logw if p ≤ i and p < i - w̃(i), else -inf. Each landmark adds a [N,H,L,L]
        # term at column p, combined by elementwise max without in-place ops.
        lmk_lb = torch.full((N, H, L, L), float("-inf"), device=device)
        k_lmk  = lmk_pos.shape[-1]
        i_idx  = idx.view(1, 1, L).float()                                         # [1,1,L] query i
        wl     = w_seq.view(N, 1, L)                                               # [N,1,L] w̃(i)
        for kk in range(k_lmk):
            pos  = lmk_pos[:, :, kk].transpose(0, 1)                               # [N,H]
            logw = lmk_logw[:, :, kk].transpose(0, 1)                              # [N,H]
            pos_c = pos.clamp(min=0)
            posf  = pos_c.float().view(N, H, 1)                                    # [N,H,1]
            ok = (pos >= 0).view(N, H, 1) & (posf <= i_idx) & (posf < (i_idx - wl))  # [N,H,L] over query i
            col  = pos_c.view(N, H, 1, 1).expand(N, H, L, 1)                       # key column j==p
            val  = torch.where(ok, logw.view(N, H, 1), torch.full((1, 1, 1), float("-inf"), device=device))
            term = torch.full((N, H, L, L), float("-inf"), device=device).scatter(
                3, col, val.unsqueeze(-1)
            )                                                                     # out-of-place
            lmk_lb = torch.maximum(lmk_lb, term)

        return torch.maximum(win_lb.unsqueeze(1), lmk_lb)                          # [N,H,L,L]

    def _packed(self, x: torch.Tensor, cu_seqlens: torch.Tensor, total_len: int, device: torch.device, rope_cs: tuple | None = None, cu_list: list | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x).view(total_len, self.n_heads,    self.head_dim)
        k = self.k_proj(x).view(total_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(total_len, self.n_kv_heads, self.head_dim)

        if rope_cs is None:
            num_seqs = cu_seqlens.shape[0] - 1
            lens     = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device)
            starts   = cu_seqlens[:-1].to(device)
            seq_ids  = torch.repeat_interleave(torch.arange(num_seqs, device=device), lens)
            pos_ids  = torch.arange(total_len, device=device) - starts[seq_ids]
            cos = self.rope_cos[pos_ids]
            sin = self.rope_sin[pos_ids]
        else:
            cos, sin = rope_cs
        # The RoPE cache is fp32 while q and k are in the autocast dtype. Casting
        # cos/sin to q's dtype keeps RoPE in the compute dtype, so the kernel receives q
        # and k in bf16 without an extra copy.
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q, k = apply_rotary_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
        # GQA: repeat the kv heads to n_heads after RoPE. Landmark selection uses
        # v_proj.weight (kv_dim wide) directly, so a landmark is selected once per kv
        # group.
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        aux = self._aux_zero(x.device, x.dtype)

        # TRAINING: differentiable path carrying the gradients of the soft window edge
        # and of the landmark weight to win_gate and alpha_w.
        # INFERENCE: the Triton (or SDPA) path below.
        if self.training:
            self._last_P = None
            # Triton training kernel: only w(i)+k keys per query, and it carries both
            # gradients.
            if _TRITON_TRAIN_OK and device.type == "cuda":
                out = self._packed_train_triton(q, k, v, x, cu_seqlens, total_len, device, cu_list)
                return self._apply_out_proj(out.view(total_len, self.d_model)), aux
            # Without Triton: dense SDPA path with the same math.
            cu   = cu_list if cu_list is not None else cu_seqlens.detach().to("cpu").tolist()
            lens = [cu[b + 1] - cu[b] for b in range(len(cu) - 1)]
            out  = self._packed_train(q, k, v, x, cu, lens, total_len, device, cu_seqlens)
            return self._apply_out_proj(out.reshape(total_len, self.d_model)), aux

        # inference (hard selectors, no gradient)
        w_sizes = self._window_sizes(x, floor=True)
        alpha   = self._alpha()

        if _TRITON_OK:
            lmk_indices, _, _ = _compute_landmark_indices(
                x.detach(), self.v_proj.weight.detach(),
                alpha.detach().float(), w_sizes,
                cu_seqlens, self.k_lmk, self.n_min, total_len, cu_list,
                head_dim=self.head_dim,
            )
            # Hard A(i) = W(i) ∪ L(i): the score only selects, with no logit bias.
            lmk_bias = torch.zeros_like(lmk_indices, dtype=torch.float32)
            out = dsalt_triton_attention(
                q, k, v, lmk_indices, lmk_bias, w_sizes, cu_seqlens,
            )
        else:
            attn_mask = _build_mask_packed(
                x, w_sizes, self.v_proj.weight.detach(),
                alpha.detach(), cu_seqlens, total_len,
                self.k_lmk, self.head_dim, device,
            )
            out = sparse_attention_forward_packed(
                q, k, v, attn_mask, self.dropout, self.training,
            )

        # The rest of this method rebuilds the dense [H,T,T] attention matrix of the
        # first sequence for the analysis metrics (rank, Dobrushin). It costs O(T^2)
        # memory, so callers that need only the output set ``_collect_last_P = False``.
        if not getattr(self, "_collect_last_P", True):
            self._last_P = None
            return self._apply_out_proj(out.view(total_len, self.d_model)), aux

        s0    = int(cu_seqlens[0])
        e0    = int(cu_seqlens[1])
        q0    = q[s0:e0].transpose(0, 1)
        k0    = k[s0:e0].transpose(0, 1)
        scale = 1.0 / math.sqrt(self.head_dim)
        T0    = e0 - s0

        cu0    = torch.tensor([0, T0], dtype=torch.int32, device=device)
        w0     = w_sizes[s0:e0]
        alpha0 = alpha.detach()

        if _TRITON_OK:
            lmk_idx, _, _ = _compute_landmark_indices(
                x[s0:e0].float().detach(),
                self.v_proj.weight.float().detach(),
                alpha0.float(),
                w0.float(),
                cu0,
                self.k_lmk,
                self.n_min,
                T0,
                [0, T0],   # host-side cu_list avoids a device-to-host sync
                head_dim=self.head_dim,
            )

            rows = torch.arange(T0, device=device)
            cols = torch.arange(T0, device=device)
            w0l  = w0.long()
            in_win = (
                (cols.unsqueeze(0) >= (rows.unsqueeze(1) - w0l.unsqueeze(1) + 1)) &
                (cols.unsqueeze(0) <= rows.unsqueeze(1))
            )

            lmk_abs   = lmk_idx[:, 0, :]
            in_lmk    = torch.zeros(self.n_heads, T0, T0, dtype=torch.bool, device=device)
            for h in range(self.n_heads):
                valid_pos = lmk_abs[h]
                # Unfilled landmark slots are padded with -1, which is not a valid index
                # (this happens when almost every token is already inside the window).
                # Mark only the valid landmark columns with index_fill_, then reapply
                # the causal cut.
                vp = valid_pos[valid_pos >= 0]                                     # [k_valid]
                if vp.numel() == 0:
                    continue
                col_is_lmk = torch.zeros(T0, dtype=torch.bool, device=device)
                col_is_lmk[vp] = True                                              # [T0] over keys
                # landmark key j is attended by query i iff j ≤ i
                in_lmk[h] = col_is_lmk.unsqueeze(0) & (cols.unsqueeze(0) <= rows.unsqueeze(1))

            full_mask = in_win.unsqueeze(0) | in_lmk
        else:
            # Without Triton: dense mask shared across heads [T0, T0]
            full_mask = _build_mask_packed(
                x[s0:e0], w0, self.v_proj.weight.detach(),
                alpha0, cu0, T0, self.k_lmk, self.head_dim, device,
            ).unsqueeze(0).expand(self.n_heads, T0, T0)

        sc        = torch.matmul(q0, k0.transpose(-2, -1)) * scale
        additive  = sc.new_full((self.n_heads, T0, T0), float("-inf")).masked_fill_(full_mask, 0.0)
        self._last_P = torch.softmax(sc + additive, dim=-1).detach()

        return self._apply_out_proj(out.view(total_len, self.d_model)), aux