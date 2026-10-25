"""Training logger: ANSI step formatter, file logging, and a throughput timer.

A custom ``logging.Formatter`` renders each training step as a colored box (loss,
ppl, representation-health metrics, window/alpha, scan cost, it/s, tok/s, VRAM),
a banner for ``training start`` and a highlighted line for validation. ``StepTimer``
provides a windowed it/s and tok/s estimate.
"""

import logging
import math
import sys
import time
from pathlib import Path

import torch

RESET    = "\033[0m"
BOLD     = "\033[1m"
DIM      = "\033[2m"
RED      = "\033[91m"
ORANGE   = "\033[38;5;208m"
GREEN    = "\033[92m"
YELLOW   = "\033[93m"
MAGENTA  = "\033[95m"
CYAN     = "\033[96m"
WHITE    = "\033[97m"
BG_GREEN = "\033[48;5;22m"
LAVENDER  = "\033[38;5;141m"
OCHRE     = "\033[38;5;179m"

_W_LABEL = 10
_W_VAL   = 14
_W_BAR   = 72


def _strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\033\[[0-9;]*m", "", s)


def _pad(text: str, width: int, align: str = ">") -> str:
    n = max(0, width - len(_strip_ansi(text)))
    return (" " * n + text) if align == ">" else (text + " " * n)


def _bar(value: float, max_val: float, width: int = 6, color: str = GREEN) -> str:
    filled = max(0, min(int(round(value / max(max_val, 1e-9) * width)), width))
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"


def _mem_color(gb: float) -> str:
    return RED if gb > 12 else (YELLOW if gb > 8 else GREEN)


def _loss_color(v: float) -> str:
    return (
        GREEN  if v < 2.0 else
        YELLOW if v < 3.5 else
        ORANGE if v < 5 else
        RED
    )


def _fmt_val(s: str, decimals: int = 4, color: str = WHITE) -> str:
    try:
        v = float(s)
        if not math.isfinite(v):
            return f"{DIM}{'nan':>{_W_VAL}}{RESET}"
        return f"{color}{v:>{_W_VAL}.{decimals}f}{RESET}"
    except Exception:
        return f"{DIM}{'nan':>{_W_VAL}}{RESET}"


def _fmt_tok_s(tok_s: float) -> str:
    """Compact throughput: 12.3M / 4.5k / 123 tok/s."""
    if tok_s >= 1e6:
        return f"{tok_s / 1e6:5.2f}M"
    if tok_s >= 1e3:
        return f"{tok_s / 1e3:5.1f}k"
    return f"{tok_s:5.0f}"


def _fmt_ppl(loss: float) -> str:
    try:
        ppl = math.exp(min(loss, 20.0))
        c   = RED if ppl > 1000 else (YELLOW if ppl > 100 else GREEN)
        return f"{c}{ppl:>{_W_VAL}.4f}{RESET}"
    except Exception:
        return f"{DIM}{'nan':>{_W_VAL}}{RESET}"


def _cell(label: str, value: str) -> str:
    l = _pad(f"{DIM}{label}{RESET}", _W_LABEL, align=">")
    v = _pad(value, _W_VAL, align=">")
    return f"{l}  {v}"


def _parse(msg: str) -> dict:
    parts = {}
    for chunk in msg.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            k, _, v = chunk.partition("=")
        else:
            sp = chunk.find(" ")
            if sp == -1:
                continue
            k, v = chunk[:sp], chunk[sp:]
        parts[k.strip()] = v.strip()
    return parts


class _StepFormatter(logging.Formatter):

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if "training start" in msg:
            return self._banner(msg)
        if "val_ppl" in msg:
            return self._val(msg)
        if "checkpoint" in msg or "done |" in msg:
            return f"  {MAGENTA}✓  {msg}{RESET}"
        # Two layouts: ``kind=cheap`` (every log_every) is one line; ``kind=metrics``
        # (every metrics_every) is the full box.
        if "kind=cheap" in msg:
            return self._step_cheap(record, msg)
        return self._step(record, msg)

    def _step_cheap(self, record: logging.LogRecord, msg: str) -> str:
        p       = _parse(msg)
        ts      = f"{DIM}{self.formatTime(record, self.datefmt)}{RESET}"
        it_s    = getattr(record, "it_s",    0.0)
        tok_s   = getattr(record, "tok_s",   0.0)
        mem_gb  = getattr(record, "mem_gb",  0.0)
        peak_gb = getattr(record, "peak_gb", 0.0)
        total_gb = getattr(record, "total_gb", 0.0)
        mem_c   = _mem_color(peak_gb if peak_gb > 0 else mem_gb)

        step = p.get("step", "?")
        try:
            loss_v = float(p.get("loss", "nan"))
            loss_s = f"{_loss_color(loss_v)}{loss_v:.4f}{RESET}"
            ppl_s  = _fmt_ppl(loss_v).strip()
        except Exception:
            loss_s = f"{DIM}nan{RESET}"
            ppl_s  = f"{DIM}nan{RESET}"
        lr_s    = f"{MAGENTA}{p.get('lr', '?')}{RESET}"
        speed_s = f"{CYAN}{it_s:5.2f} it/s{RESET}" if it_s  > 0 else f"{DIM}?.?? it/s{RESET}"
        tok_s_s = f"{CYAN}{_fmt_tok_s(tok_s)} tok/s{RESET}" if tok_s > 0 else f"{DIM}? tok/s{RESET}"
        mem_s   = (f"{mem_c}peak {peak_gb:.1f} GB{RESET}" if peak_gb > 0
                   else f"{DIM}?.? GB{RESET}")

        return (
            f"\n  {ts}  │  "
            f"{BOLD}step {CYAN}{step:<5}{RESET}  │  "
            f"loss {BOLD}{loss_s:<7}{RESET}  │  "
            f"ppl {ppl_s:<10}  │  "
            f"lr {lr_s:<8}  │  "
            f"{speed_s:<10}  │  "
            f"{tok_s_s:<13}  │  "
            f"{mem_s}"
        )

    def _banner(self, msg: str) -> str:
        bar = f"{BOLD}{CYAN}{'━' * _W_BAR}{RESET}"
        lines = [
            "",
            bar,
            f"{BOLD}{CYAN}  ██████╗   ███████╗   █████╗   ██╗    ████████╗{RESET}",
            f"{BOLD}{CYAN}  ██╔══██╗  ██╔════╝  ██╔══██╗  ██║    ╚══██╔══╝{RESET}",
            f"{BOLD}{CYAN}  ██║  ██║  ███████╗  ███████║  ██║       ██║   {RESET}",
            f"{BOLD}{CYAN}  ██║  ██║  ╚════██║  ██╔══██║  ██║       ██║   {RESET}",
            f"{BOLD}{CYAN}  ██████╔╝  ███████║  ██║  ██║  ███████╗  ██║   {RESET}",
            f"{BOLD}{CYAN}  ╚═════╝   ╚══════╝  ╚═╝  ╚═╝  ╚══════╝  ╚═╝   {RESET}",
            bar,
        ]
        for part in msg.split("|"):
            part = part.strip()
            if not part:
                continue
            k, _, v = part.partition("=")
            lines.append(f"  {YELLOW}{k.strip():>22}{RESET}  {DIM}={RESET}  {LAVENDER}{v.strip()}{RESET}")
        lines.append(bar)
        lines.append("")
        return "\n".join(lines)

    def _val(self, msg: str) -> str:
        is_best = "← best" in msg
        c       = GREEN if is_best else YELLOW
        badge   = f"  {BOLD}{GREEN}  ★ NEW BEST  {RESET}" if is_best else ""
        bar     = f"{c}{'─' * 60}{RESET}"
        return f"\n{bar}\n  {BOLD}{c}VAL{RESET}  {LAVENDER}{msg}{RESET}{badge}\n{bar}\n"

    def _step(self, record: logging.LogRecord, msg: str) -> str:
        p      = _parse(msg)
        ts     = f"{DIM}{self.formatTime(record, self.datefmt)}{RESET}"
        it_s    = getattr(record, "it_s",    0.0)
        tok_s   = getattr(record, "tok_s",   0.0)
        mem_gb  = getattr(record, "mem_gb",  0.0)
        peak_gb = getattr(record, "peak_gb", 0.0)
        mem_c   = _mem_color(peak_gb if peak_gb > 0 else mem_gb)

        step = p.get("step", "?")

        try:
            loss_v = float(p.get("loss", "nan"))
            lc     = _loss_color(loss_v)
            loss_s = f"{lc}{loss_v:>{_W_VAL}.4f}{RESET}"
            ppl_s  = _fmt_ppl(loss_v)
        except Exception:
            loss_s = f"{DIM}{'nan':>{_W_VAL}}{RESET}"
            ppl_s  = f"{DIM}{'nan':>{_W_VAL}}{RESET}"

        sigma2 = _fmt_val(p.get("σ²",      p.get("sigma2", "nan")), color=YELLOW)
        rank   = _fmt_val(p.get("rank",     "nan"),                  color=CYAN)
        res    = _fmt_val(p.get("res",      "nan"),                  color=OCHRE)
        H      = _fmt_val(p.get("H",        "nan"),                  color=GREEN)
        noise  = _fmt_val(p.get("noise",    "nan"),                  color=RED)
        sink   = _fmt_val(p.get("sink",     "nan"),                  color=YELLOW)
        hstd   = _fmt_val(p.get("head_std", p.get("hstd", "nan")),   color=MAGENTA)
        tdist  = _fmt_val(p.get("token_dist", "nan"),                color=LAVENDER)
        lr_s   = f"{MAGENTA}{p.get('lr', '?')}{RESET}"
        # window and alpha as preformatted min/mean/max triplets
        win_s   = f"{CYAN}{p.get('win',   'nan'):>{_W_VAL}}{RESET}"
        alpha_s = f"{GREEN}{p.get('alpha', 'nan'):>{_W_VAL}}{RESET}"
        # kernel key-block scan cost
        scan_s  = f"{RED}{p.get('scan',  'nan'):>{_W_VAL}}{RESET}"

        speed_s = f"{CYAN}{it_s:6.2f} it/s{RESET}" if it_s   > 0 else f"{DIM}  ?.?? it/s{RESET}"
        tok_s_s = f"{CYAN}{_fmt_tok_s(tok_s)} tok/s{RESET}" if tok_s > 0 else f"{DIM}  ? tok/s{RESET}"
        total_gb = getattr(record, "total_gb", 0.0)
        mem_s = (
            f"{mem_c}cur {mem_gb:.2f}{RESET}{DIM}·{RESET}{mem_c}peak {peak_gb:.2f} GB{RESET}{DIM} / {total_gb:.0f} tot{RESET}"
            if peak_gb > 0 else f"{DIM}  ?.? GB{RESET}"
        )
        mem_bar = _bar(mem_gb, 16.0, width=6, color=mem_c)

        sep = f"{DIM}{'─' * _W_BAR}{RESET}"
        G   = "    "

        try:
            total    = int(p.get("total", 0))
            step_int = int(step)
            progress = f"{BOLD}step {CYAN}{step_int}{RESET}{DIM}/{RESET}{CYAN}{total}{RESET}" if total else f"{BOLD}step {CYAN}{step}{RESET}"
        except Exception:
            progress = f"{BOLD}step {CYAN}{step}{RESET}"

        hdr = f"  {ts}   {progress}   {speed_s}   {tok_s_s}   {mem_s}"

        rows = [
            f"  {_cell('loss', loss_s)}{G}{_cell('ppl',       ppl_s)}",
            f"  {_cell('σ²',   sigma2)}{G}{_cell('rank',      rank)}",
            f"  {_cell('residual', res)}{G}{_cell('entropy H', H)}",
            f"  {_cell('noise', noise)}{G}{_cell('sink',      sink)}",
            f"  {_cell('head_std', hstd)}{G}{_cell('lr',      lr_s)}",
            f"  {_cell('win μ', win_s)}{G}{_cell('alpha μ', alpha_s)}",
            f"  {_cell('scan', scan_s)}{G}{_cell('token_dist', tdist)}",
        ]

        return "\n".join(["", sep, hdr, sep] + rows + [sep])


class _DSALTFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        for field in ["it_s", "tok_s", "mem_gb", "peak_gb", "rank_eff", "total_gb"]:
            if not hasattr(record, field):
                setattr(record, field, 0.0)
        return True


def get_logger(name: str, log_dir: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Build (once) the DSALT logger: colored stdout + optional ``train.log`` file.

    Idempotent, returns the existing logger if it already has handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_StepFormatter(datefmt="%H:%M:%S"))
    sh.addFilter(_DSALTFilter())
    logger.addHandler(sh)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / "train.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
        fh.addFilter(_DSALTFilter())
        logger.addHandler(fh)

    logger.propagate = False
    return logger


class StepTimer:
    """Windowed step timer: a rolling-average it/s and instantaneous tok/s."""

    def __init__(self, window: int = 50, device: torch.device = None):
        self._window = window
        self._times: list[float] = []
        self._t0: float | None   = None
        self._device = device

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self, total_tokens: int = 0) -> dict:
        if self._t0 is None:
            return {}
        elapsed = time.perf_counter() - self._t0
        self._times.append(elapsed)
        if len(self._times) > self._window:
            self._times.pop(0)
        avg_time = sum(self._times) / len(self._times)
        return {
            "it_s":      1.0 / avg_time if avg_time > 0 else 0.0,
            "tok_s":     total_tokens / elapsed if elapsed > 0 else 0.0,
            "step_time": elapsed,
        }