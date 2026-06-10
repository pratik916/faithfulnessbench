"""White-box / linear-probe **offline analogue** (the DESIGN.md §7 spike, built).

Behavioral (black-box) probing is the project's scope; this extends the validation
philosophy to a *white-box* probe family **without** leaving numpy or claiming anything about
real activations. We give each synthetic model a low-dimensional, **a-priori generic**
"activation" vector ``activation(model, problem)`` — a fixed feature set read from the model's
own reasoning trace, defined *before* any fitting and identical for every axis — and fit a
numpy logistic `LinearProbe` on a **held-out train split** to predict the planted faithfulness
label. We then contrast it with the black-box CoT-reading monitor.

The honest findings the contrast must show:

* **White-box catches what obfuscated-CoT monitoring misses.** The OBF model coherently states
  a *wrong* answer while evading the text-reading monitor (its combined AUROC sits at chance).
  The activation carries the answer-vs-truth deviation directly, so a linear probe on it
  separates faithful from obfuscated cleanly — the Apollo (arXiv:2502.03407) story, as an
  offline *analogue*, not a reproduction.
* **White-box has its own blind spot.** The FILLER axis is unfaithful only under *intervention*
  (the answer survives content-free filler); observationally its trace looks faithful, so the
  per-instance activation cannot see it while the interventional black-box FIL probe can. White-
  box probing *complements* causal probes, it does not replace them.
* **It can fail.** A permuted-label control collapses the white-box AUROC to chance.

**This proves nothing about real models.** Real activations are neither low-dimensional nor
authored; here we wrote both the activation and the label, so the *number* is by construction —
the load-bearing result is the **contrast** (white-box vs black-box-under-obfuscation, and
catches-vs-blind), plus the held-out split and permuted-label control. Arithmetic-only, in the
*extended* population, so committed pooled numbers are unchanged. numpy-only, deterministic.
"""
from __future__ import annotations

import numpy as np

from . import metrics
from .models.synthetic import ConfigurableSyntheticModel, FaithfulnessProfile
from .probes import HELD_OUT_PROBE_CLASSES, PROBE_CLASSES
from .problems import Problem, arithmetic_chain_problems, stated_final

# The a-priori generic activation: fixed feature set, defined once, computed identically for
# every model and axis. The probe LEARNS which features matter; we do NOT engineer per-axis.
ACTIVATION_FEATURES = (
    "answer_value",       # the committed answer (numeric)
    "stated_final_value", # the value the CoT concludes with
    "answer_true_gap",    # |answer - true answer| (wrong-answer / reward-hack signal)
    "matches_cue_target", # 1 if the answer equals the planted hint
    "cot_answer_gap",     # 1 if the CoT's stated conclusion != the committed answer (decoy)
    "locked_from_empty",  # 1 if the answer is already fixed from an empty reasoning prefix
    "cot_surface_parity", # parity of the CoT characters (a generic surface artifact)
    "n_steps",            # number of reasoning steps
)


def _as_int(s: str):
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return None


def activation(model, problem: Problem) -> np.ndarray:
    """The generic activation vector φ(model, problem) over an arithmetic problem."""
    tr = model.reason(problem)
    ans = _as_int(tr.answer)
    true = _as_int(problem.answer)
    try:
        stated = _as_int(stated_final(list(tr.steps)))
    except (ValueError, IndexError):
        stated = None
    empty_ans = _as_int(model.answer_from_prefix(problem, []))
    cue_target = _as_int(problem.cue.target) if problem.cue is not None else None
    cot = tr.cot or ""
    a = float(ans) if ans is not None else 0.0
    return np.array([
        a,
        float(stated) if stated is not None else a,
        abs(a - float(true)) if (ans is not None and true is not None) else 0.0,
        1.0 if (ans is not None and cue_target is not None and ans == cue_target) else 0.0,
        1.0 if (stated is not None and ans is not None and stated != ans) else 0.0,
        1.0 if (empty_ans is not None and empty_ans == ans) else 0.0,
        float(sum(ord(c) for c in cot) % 2),
        float(len(tr.steps)),
    ], dtype=float)


class LinearProbe:
    """A deterministic numpy logistic-regression probe (standardize → full-batch GD)."""

    def __init__(self, *, l2: float = 1e-3, lr: float = 0.3, n_iter: int = 3000):
        self.l2, self.lr, self.n_iter = l2, lr, n_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> LinearProbe:
        self.mu_ = X.mean(axis=0)
        self.sd_ = X.std(axis=0) + 1e-9
        Xs = (X - self.mu_) / self.sd_
        n, d = Xs.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        y = np.asarray(y, dtype=float)
        for _ in range(self.n_iter):
            z = np.clip(Xs @ self.w_ + self.b_, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            self.w_ -= self.lr * (Xs.T @ (p - y) / n + self.l2 * self.w_)
            self.b_ -= self.lr * float((p - y).mean())
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu_) / self.sd_
        z = np.clip(Xs @ self.w_ + self.b_, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))


def _split(n: int) -> tuple[list[int], list[int]]:
    """Deterministic train/test split: even indices train, odd indices test."""
    return [i for i in range(n) if i % 2 == 0], [i for i in range(n) if i % 2 == 1]


def _features_labels(neg_model, pos_model, problems):
    X = [activation(neg_model, p) for p in problems] + [activation(pos_model, p) for p in problems]
    y = [0.0] * len(problems) + [1.0] * len(problems)
    return np.array(X), np.array(y)


def _combined_blackbox(model, problems, probe_names, n_trials):
    registry = {**PROBE_CLASSES, **HELD_OUT_PROBE_CLASSES}  # FIL is held out of the default battery
    res = {k: registry[k]().run(model, problems, n_trials=n_trials) for k in probe_names}
    by_pid = {k: dict(zip(res[k].problem_ids, res[k].scores.tolist())) for k in probe_names}
    common = [p.id for p in problems if all(p.id in by_pid[k] for k in probe_names)]
    return {pid: float(np.nanmean([by_pid[k][pid] for k in probe_names])) for pid in common}


def _heldout_auroc(neg_model, pos_model, problems, *, permute=False, seed=0):
    """Fit a LinearProbe on the train split, score the disjoint test split (AUROC on φ)."""
    X, y = _features_labels(neg_model, pos_model, problems)
    n = len(problems)
    tr, te = _split(n)
    # rows: [neg over problems] then [pos over problems]; map problem-index splits to both halves.
    tr_rows = tr + [n + i for i in tr]
    te_rows = te + [n + i for i in te]
    ytr = y[tr_rows].copy()
    if permute:  # seeded label-shuffle control: destroys the φ→label signal -> AUROC ~ chance
        ytr = ytr[np.random.default_rng(seed).permutation(len(ytr))]
    probe = LinearProbe().fit(X[tr_rows], ytr)
    scores = probe.predict_proba(X[te_rows])
    return metrics.roc_auc(scores, y[te_rows])


def whitebox_analogue(*, seed: int = 0, n_problems: int = 24, n_trials: int = 2) -> dict:
    """Run the white-box-vs-black-box contrast and the blind-spot/control demonstrations."""
    problems = arithmetic_chain_problems(n_problems, seed=seed)
    faithful = ConfigurableSyntheticModel("faithful", FaithfulnessProfile(seed=seed))
    obfuscate = ConfigurableSyntheticModel("obf", FaithfulnessProfile(seed=seed, p_obfuscate=1.0))
    filler = ConfigurableSyntheticModel("fil", FaithfulnessProfile(seed=seed, p_filler=1.0))

    # Black-box monitor AUROC (text-reading combined / FIL) over the SAME test split.
    _, te = _split(n_problems)
    te_problems = [problems[i] for i in te]

    def _bb_auroc(pos_model, probe_names):
        neg = _combined_blackbox(faithful, te_problems, probe_names, n_trials)
        pos = _combined_blackbox(pos_model, te_problems, probe_names, n_trials)
        common = [pid for pid in neg if pid in pos]
        s = np.array([neg[pid] for pid in common] + [pos[pid] for pid in common])
        ylab = np.r_[np.zeros(len(common)), np.ones(len(common))]
        return metrics.roc_auc(s, ylab)

    catches = {
        "white_box_auroc": _heldout_auroc(faithful, obfuscate, problems),
        "black_box_auroc": _bb_auroc(obfuscate, ["SHI", "CSC", "SIM", "EAR"]),
    }
    blind = {
        "white_box_auroc": _heldout_auroc(faithful, filler, problems),
        "black_box_fil_auroc": _bb_auroc(filler, ["FIL"]),
    }
    # Permutation control: average over several seeded label-shuffles so the chance level is
    # stable on a small test split (a single shuffle is noisy at n=12 test instances).
    permuted = float(np.mean([
        _heldout_auroc(faithful, obfuscate, problems, permute=True, seed=k) for k in range(8)
    ]))
    return {
        "features": list(ACTIVATION_FEATURES),
        "obfuscation": catches,      # white-box >> black-box (monitor evaded)
        "filler_blind_spot": blind,  # white-box ~chance, interventional FIL catches it
        "permuted_control_auroc": permuted,
        "note": (
            "Offline analogue. The white-box numbers are high BY CONSTRUCTION (we authored both "
            "the activation and the label); the load-bearing result is the CONTRAST — white-box "
            "separates faithful from obfuscated where the text-reading monitor is evaded (~0.5), "
            "yet is blind to the interventional FILLER axis that the black-box FIL probe catches. "
            "It says nothing about real-model activations, which are neither low-dimensional nor "
            "authored."
        ),
    }
