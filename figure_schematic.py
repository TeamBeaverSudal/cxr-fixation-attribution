"""Rebuild the architecture figure with (a) the label embedding on the query only and
(b) a real predicted heatmap instead of an illustration.

The label embedding is also concatenated into the key in code, but its contribution is
provably inert: it is identical for every fixation, so it shifts every score equally and
the softmax removes it (measured gradient 1.5e-8 vs 1.8e-1 for the position columns).
Drawing it into the key would misdescribe the mechanism, so it is drawn into the query.

  python figure_schematic.py            # -> fig1.{svg,pdf,png}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from core import raster, iou, word_feat, WORD_DIM
from selector import train_model, predict_raw, blur_norm, POS_DIM
from evaluate import FUSION, POS_MODE

GAZE, LANG, ATTN, OUT = "#d98324", "#1b7f79", "#6b4fa8", "#c94f4f"
INK, MUTE, PANEL = "#1a1a1a", "#6b6b6b", "#faf8f5"


def real_heatmap(cache="align.pt", epochs=40, seed=0):
    """Train the shipped configuration and return the median-IoU test instance's heatmap.
    Median, not best -- a figure picked for looking good is worse than no figure."""
    import torch
    recs, labels = torch.load(cache, weights_only=False)
    li = {L: i for i, L in enumerate(labels)}
    subs = np.array([r["subject"] for r in recs]); uniq = np.array(sorted(set(subs)))
    rng = np.random.default_rng(0); rng.shuffle(uniq); n = len(uniq)
    val_s = set(uniq[:int(n * .15)]); test_s = set(uniq[int(n * .15):int(n * .45)])
    part = lambda r: "val" if r["subject"] in val_s else "test" if r["subject"] in test_s else "train"

    def insts(p, keys=False):
        out = []
        for r in recs:
            if part(r) != p:
                continue
            for d in r["labels"]:
                item = (r["fix"], d["mentions"], d["ellipses"], li[d["label"]],
                        word_feat(d.get("mtext", [])))
                out.append(item + ((r["rid"], d["label"]),) if keys else item)
        return out
    tr, te = insts("train"), insts("test", keys=True)
    print(f"train/test = {len(tr)}/{len(te)}", flush=True)

    net = train_model(tr, labels, use_position=True, epochs=epochs, use_text=True,
                      fusion=FUSION, pos_mode=POS_MODE, seed=seed)
    SIGMA, THR = 2.5, 0.012      # the validation-tuned operating point for this config
    hm, sc = [], []
    for f, m, e, l, w, k in te:
        h = blur_norm(predict_raw(net, f, m, l, True, use_text=True, wf=w,
                                  pos_mode=POS_MODE), SIGMA)
        hm.append(h); sc.append(iou(h, raster(e), THR))
    sc = np.array(sc)
    j = int(np.nanargmin(np.abs(sc - np.nanmedian(sc))))    # instance nearest the median
    f, m, e, l, w, key = te[j]
    print(f"median IoU {np.nanmedian(sc):.4f}; picked {key} IoU={sc[j]:.4f}", flush=True)
    return hm[j], raster(e), f, sc[j], key


def box(ax, x, y, w, h, label, sub=None, c=INK, fill="white", lw=1.4, fs=9.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fill, ec=c, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label, ha="center", va="center",
            fontsize=fs, color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=fs - 2, color=MUTE, style="italic", zorder=3)


def arrow(ax, p, q, c=MUTE, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=c, lw=lw, zorder=1,
                                 shrinkA=2, shrinkB=2))


def build(hm, gt, fix, iou_val, key, out="fig1"):
    fig, ax = plt.subplots(figsize=(13.2, 6.0))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    # ---- gaze track (top) --------------------------------------------------
    ax.text(0.012, 0.955, "GAZE TRACK", fontsize=9, color=GAZE, weight="bold")
    ax.text(0.012, 0.915, "per fixation, N per instance", fontsize=8, color=MUTE, style="italic")
    box(ax, 0.012, 0.70, 0.175, 0.175, "Temporal / kinematic",
        "5-dim:  Δt, |Δt|, before,\nduration, velocity", c=GAZE, fill=PANEL)
    box(ax, 0.012, 0.50, 0.175, 0.155, "Fourier position",
        f"{POS_DIM}-dim:  (x, y), L = 4", c=GAZE, fill=PANEL)
    box(ax, 0.235, 0.585, 0.115, 0.185, "Linear", "→ key  $k_i$", c=GAZE)
    arrow(ax, (0.187, 0.787), (0.235, 0.705), GAZE)
    arrow(ax, (0.187, 0.578), (0.235, 0.652), GAZE)
    ax.text(0.293, 0.545, f"$K \\in \\mathbb{{R}}^{{N \\times 16}}$", ha="center",
            fontsize=9, color=GAZE)

    # ---- language track (bottom) -------------------------------------------
    ax.text(0.012, 0.395, "LANGUAGE TRACK", fontsize=9, color=LANG, weight="bold")
    ax.text(0.012, 0.355, "one per instance", fontsize=8, color=MUTE, style="italic")
    box(ax, 0.012, 0.155, 0.175, 0.165, "Report descriptor",
        f"{WORD_DIM}-dim:  7 spatial\n+ 3 severity terms", c=LANG, fill=PANEL)
    box(ax, 0.012, 0.015, 0.175, 0.105, "Finding label", "8-dim embedding", c=LANG, fill=PANEL)
    box(ax, 0.235, 0.105, 0.115, 0.185, "Linear", "→ query  $q$", c=LANG)
    arrow(ax, (0.187, 0.238), (0.235, 0.225), LANG)
    arrow(ax, (0.187, 0.068), (0.235, 0.160), LANG)
    ax.text(0.293, 0.065, "$q \\in \\mathbb{R}^{16}$", ha="center", fontsize=9, color=LANG)

    # label-embedding note -- the correction this revision exists for
    ax.text(0.012, -0.045, "the label is concatenated into both, so the key input is 29-dim; in the key it is\n"
                           "constant across fixations, so the softmax cancels it and only the query conditions",
            fontsize=7.5, color=MUTE, style="italic", va="top")

    # ---- attention ---------------------------------------------------------
    box(ax, 0.405, 0.30, 0.185, 0.40, "", c=ATTN, fill="#f7f4fc", lw=1.6)
    ax.text(0.4975, 0.645, "CROSS-ATTENTION", ha="center", fontsize=9,
            color=ATTN, weight="bold")
    ax.text(0.4975, 0.605, "single layer, no value projection", ha="center",
            fontsize=7.5, color=MUTE, style="italic")
    ax.text(0.4975, 0.485, r"$\alpha = \mathrm{softmax}\!\left(\dfrac{K q}{\sqrt{d_k}}\right)$",
            ha="center", va="center", fontsize=15, color=INK)
    ax.text(0.4975, 0.355, "softmax over the N fixations", ha="center",
            fontsize=8, color=MUTE)
    arrow(ax, (0.350, 0.660), (0.405, 0.560), GAZE, lw=1.6)
    arrow(ax, (0.350, 0.190), (0.405, 0.430), LANG, lw=1.6)

    # ---- splat -------------------------------------------------------------
    box(ax, 0.635, 0.42, 0.145, 0.22, "Splat onto real\nfixation positions",
        "α weights the observed (x, y)", c=OUT, fill=PANEL)
    arrow(ax, (0.590, 0.500), (0.635, 0.520), ATTN, lw=1.6)
    box(ax, 0.635, 0.235, 0.145, 0.105, "Gaussian blur\n+ normalize", None, c=OUT)
    arrow(ax, (0.7075, 0.420), (0.7075, 0.340), OUT)

    # ---- real predicted heatmap -------------------------------------------
    axh = fig.add_axes([0.815, 0.30, 0.165, 0.46])
    axh.imshow(hm, cmap="inferno", origin="upper")
    axh.contour(np.linspace(0, hm.shape[1] - 1, gt.shape[1]),
                np.linspace(0, hm.shape[0] - 1, gt.shape[0]),
                gt.astype(float), levels=[0.5], colors="#3ddc84", linewidths=1.6)
    axh.set_xticks([]); axh.set_yticks([])
    for sp in axh.spines.values():
        sp.set_edgecolor(OUT); sp.set_linewidth(1.4)
    axh.set_title("Predicted heatmap", fontsize=9.5, color=INK, pad=6)
    axh.set_xlabel(f"real model output, median-IoU test case (IoU = {iou_val:.3f})\n"
                   "green outline = ground-truth ellipse",
                   fontsize=7.5, color=MUTE, style="italic", labelpad=5)
    arrow(ax, (0.782, 0.520), (0.806, 0.520), OUT)

    fig.savefig(f"{out}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{out}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}.svg / .pdf / .png   (case {key}, IoU {iou_val:.4f})")


if __name__ == "__main__":
    hm, gt, fix, v, key = real_heatmap()
    assert hm.max() > 0 and np.isfinite(hm).all(), "heatmap degenerate"
    build(hm, gt, fix, v, key)
