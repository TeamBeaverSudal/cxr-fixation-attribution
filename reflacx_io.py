"""REFLACX raw gaze -> spatiotemporal representations.

Idea under test (team-sudal.tistory.com/192): the raw 1000 Hz sample stream,
kept as a spatiotemporal trajectory, is a stronger weak-supervision signal
than parsed fixations. Parsing discards saccade paths, velocity profiles and
micro-motions; here we keep them.

REFLACX already maps gaze to original-image pixel coordinates (blinks = NaN),
so no screen->image calibration is needed. All maps are built on a downscaled
grid (max side ~512 px) shared with the lesion-ellipse mask.

Representations (naming follows the blog post):
  A  fixation_heatmap  : duration-weighted Gaussian density   (GazeMedSeg-style)
  B  fixation_sequence : parsed (x, y, onset, dur) events     (GradTrack-style)
  C  raw_map / raw_clip: every raw sample splatted, optionally weighted by
                         slowness exp(-v/tau) or speech proximity  (the new one)

Training-free metrics vs anomaly ellipses: budgeted IoU (top-|mask| pixels),
mass-in-lesion, time-to-first-hit.

Run `python reflacx_io.py` for a synthetic self-check (no dataset needed).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

RECORD_FILES = ("gaze.csv", "fixations.csv", "anomaly_location_ellipses.csv",
                "timestamps_transcription.csv", "transcription.txt")

LESION_TERMS = ("nodule", "mass", "opacit", "consolidat", "effusion",
                "pneumothorax", "atelecta", "edema", "cardiomegaly",
                "fracture", "emphysema", "thicken", "infiltrat")


# ---------------------------------------------------------------- data access

def find_records(root):
    """{reflacx_id: {filename: path}} for dirs holding gaze or fixation csvs.

    Handles both <root>/main_data/<id>/... and <root>/<id>/... ;
    raw gaze may live in a sibling gaze_data/<id>/gaze.csv (official layout).
    """
    recs = {}
    for pat in ("*", "*/*"):
        for d in Path(root).glob(pat):
            if not d.is_dir():
                continue
            present = {f: d / f for f in RECORD_FILES if (d / f).exists()}
            if "fixations.csv" in present or "gaze.csv" in present:
                recs.setdefault(d.name, {}).update(present)
    return recs


def _col(df, *cands):
    for c in cands:
        if c in df.columns:
            return c
    raise KeyError(f"none of {cands} found in columns {list(df.columns)}")


def _df(x):
    return x if isinstance(x, pd.DataFrame) else pd.read_csv(x)


def _seconds(t):
    # ponytail: unit guard — a reading spans ~10-120 s; >3600 means milliseconds
    return t / 1000.0 if len(t) > 1 and t[-1] - t[0] > 3600 else t


def _velocity(gz):
    t, x, y, valid = gz["t"], gz["x"], gz["y"], gz["valid"]
    rx, ry = gz["ppd"]
    v = np.full(len(t), np.nan)
    if len(t) < 2:
        return v
    dt = np.diff(t)
    step = np.hypot(np.diff(x) / rx, np.diff(y) / ry)  # degrees
    ok = valid[1:] & valid[:-1] & (dt > 0) & (dt < 0.05)  # no v across blinks/gaps
    v[1:][ok] = step[ok] / dt[ok]
    return v


# ------------------------------------------------------------ representations

def grid_shape(img_h, img_w, max_side=512):
    s = max_side / max(img_h, img_w)
    return (round(img_h * s), round(img_w * s)), s


def splat(x, y, w, shape, scale, sigma_px):
    """Weighted point masses -> blurred density map.

    ponytail: one global blur instead of per-point kernels; angular resolution
    varies <10% within a record, and this is ~100x faster.
    """
    H, W = shape
    x, y, w = (np.asarray(a, float) for a in (x, y, w))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    xi = (x[keep] * scale).astype(int)
    yi = (y[keep] * scale).astype(int)
    wk = w[keep]
    inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)  # drop off-image gaze
    m = np.zeros(shape, np.float32)
    np.add.at(m, (yi[inb], xi[inb]), wk[inb])
    return gaussian_filter(m, sigma_px)


def _sigma_px(ppd, scale, deg):
    return max(0.5, np.mean(ppd) * scale * deg)


def fixation_heatmap(fix, shape, scale, sigma_deg=1.0):
    """A: duration-weighted fixation density (the GazeMedSeg-style baseline)."""
    f = _df(fix)
    x = f[_col(f, "x_position", "average_x_position")].to_numpy(float)
    y = f[_col(f, "y_position", "average_y_position")].to_numpy(float)
    t0 = _seconds(f[_col(f, "timestamp_start_fixation")].to_numpy(float))
    t1 = _seconds(f[_col(f, "timestamp_end_fixation")].to_numpy(float))
    try:
        ppd = (np.nanmedian(f[_col(f, "angular_resolution_x_pixels_per_degree")]),
               np.nanmedian(f[_col(f, "angular_resolution_y_pixels_per_degree")]))
    except KeyError:
        ppd = (50.0, 50.0)
    return splat(x, y, t1 - t0, shape, scale, _sigma_px(ppd, scale, sigma_deg))


def fixation_sequence(fix):
    """B: (N, 4) array of x, y, onset, duration — input for sequence models."""
    f = _df(fix)
    x = f[_col(f, "x_position", "average_x_position")].to_numpy(float)
    y = f[_col(f, "y_position", "average_y_position")].to_numpy(float)
    t0 = _seconds(f[_col(f, "timestamp_start_fixation")].to_numpy(float))
    t1 = _seconds(f[_col(f, "timestamp_end_fixation")].to_numpy(float))
    return np.stack([x, y, t0 - t0.min(), t1 - t0], axis=1)


def w_uniform(gz):
    return np.ones(len(gz["t"]))


def w_slow(gz, tau=30.0):
    """Slowness weight exp(-v/tau): continuous stand-in for 'is a fixation'.

    Keeps drift / smooth pursuit / fixation micro-motion, suppresses saccades.
    tau in deg/s; ~30 sits between fixational drift (<20) and saccades (>100).
    """
    return np.exp(-np.nan_to_num(gz["v"]) / tau)


def w_fast(gz, tau=30.0):
    """Saccade-emphasis weight 1 - exp(-v/tau): the transit paths parsing
    throws away. Saccades sweep lesion borders -> boundary prior (idea 2).
    Score this map with mass_near_boundary, not iou."""
    return 1.0 - np.exp(-np.nan_to_num(gz["v"]) / tau)


def w_pupil(gz, lag=0.75):
    """Cognitive-load weight from lagged, z-scored normalized pupil area
    (idea 4). Pupil dilation trails its driving fixation by ~0.5-1 s, so we
    shift the signal back onto the sample that caused it. Uniform if no pupil."""
    p = gz.get("pupil")
    if p is None or not np.isfinite(p).any():
        return w_uniform(gz)
    dt = np.median(np.diff(gz["t"])) or 1e-3
    pl = np.roll(p, -int(round(lag / dt)))  # dilation at t -> cause at t-lag
    z = (pl - np.nanmean(pl)) / (np.nanstd(pl) + 1e-9)
    return np.clip(np.nan_to_num(z), 0, None)  # only above-average load counts


def w_revisit(gz, scale, cell_deg=2.0, gap=2.0):
    """Late-revisit weight (idea 3): a sample's weight is how many times its
    ~cell_deg region has been re-entered after being left for >gap seconds.
    Radiologists return to verify suspects; first-pass scanning stays at 0.

    ponytail: O(n) dict walk over samples — a few M ops per record, fine.
    """
    cell = max(1.0, np.mean(gz["ppd"]) * cell_deg * scale)
    t, valid = gz["t"], gz["valid"]
    cx = (np.nan_to_num(gz["x"]) * scale / cell).astype("int64")
    cy = (np.nan_to_num(gz["y"]) * scale / cell).astype("int64")
    last_t, visits, w = {}, {}, np.zeros(len(t))
    for i in range(len(t)):
        if not valid[i]:
            continue
        key = (cx[i], cy[i])
        prev = last_t.get(key)
        if prev is None or t[i] - prev > gap:
            visits[key] = visits.get(key, 0) + 1  # a fresh visit to this cell
        last_t[key] = t[i]
        w[i] = visits[key] - 1  # 0 on first visit, +1 per return, whole dwell
    return w


def w_speech(gz, words, terms=LESION_TERMS, lead=2.5):
    """Weight samples in the `lead` seconds before a lesion word is spoken.

    Zero everywhere if no term matches (caller should skip the map then).
    """
    wd = _df(words)
    tw = _seconds(wd[_col(wd, "timestamp_start_word", "start", "timestamp_start")]
                  .to_numpy(float))
    said = wd[_col(wd, "word")].astype(str).str.lower()
    hits = tw[said.str.contains("|".join(terms), na=False, regex=True)]
    w = np.zeros(len(gz["t"]))
    for h in hits:
        w[(gz["t"] >= h - lead) & (gz["t"] < h)] += 1.0
    return w


def raw_map(gz, shape, scale, weights=None, sigma_deg=0.5):
    """C (collapsed): every raw sample splatted with optional weights."""
    w = w_uniform(gz) if weights is None else weights
    return splat(gz["x"], gz["y"], np.where(gz["valid"], w, np.nan),
                 shape, scale, _sigma_px(gz["ppd"], scale, sigma_deg))


def raw_clip(gz, shape, scale, T=32, weights=None, sigma_deg=0.5):
    """C (video): (T, H, W) clip replaying the scanpath, for 3D models / viz."""
    H, W = shape
    t, valid = gz["t"], gz["valid"]
    w = w_uniform(gz) if weights is None else np.asarray(weights, float)
    span = max(t[-1] - t[0], 1e-9)
    fi = np.minimum(((t - t[0]) / span * T).astype(int), T - 1)
    clip = np.zeros((T, H, W), np.float32)
    xi = np.full(len(t), -1)
    yi = np.full(len(t), -1)
    xi[valid] = (gz["x"][valid] * scale).astype(int)
    yi[valid] = (gz["y"][valid] * scale).astype(int)
    inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & np.isfinite(w)
    np.add.at(clip, (fi[inb], yi[inb], xi[inb]), w[inb])
    s = _sigma_px(gz["ppd"], scale, sigma_deg)
    return gaussian_filter(clip, (0, s, s))


# ----------------------------------------------------------- ground truth / metrics

def ellipses_mask(ellipses, shape, scale, min_certainty=0):
    """Union of inscribed ellipses from anomaly_location_ellipses.csv."""
    e = _df(ellipses)
    if "certainty" in e.columns:
        e = e[e["certainty"] >= min_certainty]
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    m = np.zeros(shape, bool)
    for _, r in e.iterrows():
        cx, cy = (r["xmin"] + r["xmax"]) / 2 * scale, (r["ymin"] + r["ymax"]) / 2 * scale
        a, b = (r["xmax"] - r["xmin"]) / 2 * scale, (r["ymax"] - r["ymin"]) / 2 * scale
        if a > 0 and b > 0:
            m |= ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
    return m


def iou_budget(m, mask):
    """IoU of the top-|mask| pixels of m vs mask. Parameter-free: every
    representation spends the same pixel budget, so maps are comparable."""
    k = int(mask.sum())
    if k == 0 or m.sum() <= 0:
        return np.nan
    thr = np.partition(m.ravel(), -k)[-k]
    pred = m >= thr
    inter = (pred & mask).sum()
    return inter / (pred.sum() + k - inter)


def mass_in(m, mask):
    """Fraction of map mass inside the lesion mask."""
    tot = m.sum()
    return float((m * mask).sum() / tot) if tot > 0 else np.nan


def mass_near_boundary(m, mask, band_deg=None, band_px=None):
    """Fraction of map mass within a band straddling the lesion contour.
    Tests whether a map concentrates on borders (idea 2) vs centers/anywhere."""
    from scipy.ndimage import distance_transform_edt
    if mask.sum() == 0 or m.sum() <= 0:
        return np.nan
    if band_px is None:
        band_px = max(3, int(0.03 * max(mask.shape)))
    d_out = distance_transform_edt(~mask)  # 0 inside, grows outward
    d_in = distance_transform_edt(mask)    # 0 outside, grows inward
    band = ((d_out > 0) & (d_out <= band_px)) | ((d_in > 0) & (d_in <= band_px))
    return float((m * band).sum() / m.sum())


def time_to_first_hit(gz, mask, scale):
    """Seconds from first valid sample until gaze first lands in the mask."""
    H, W = mask.shape
    t, valid = gz["t"], gz["valid"]
    xi = np.where(valid, gz["x"], -1)
    yi = np.where(valid, gz["y"], -1)
    xi = (xi * scale).astype(int)
    yi = (yi * scale).astype(int)
    inb = valid & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    hit = np.zeros(len(t), bool)
    hit[inb] = mask[yi[inb], xi[inb]]
    if not valid.any() or not hit.any():
        return np.nan
    return float(t[hit.argmax()] - t[valid.argmax()])


# ------------------------------------------------------------------ self-check

def _selfcheck():
    rng = np.random.default_rng(0)
    img_h, img_w, ppd = 800, 1000, 50.0
    shape, scale = grid_shape(img_h, img_w)
    ell = pd.DataFrame([{"xmin": 250, "ymin": 330, "xmax": 450, "ymax": 490,
                         "certainty": 5}])
    mask = ellipses_mask(ell, shape, scale)

    # 8 s slow dwell in lesion, 6 s fast sweeps, 6 s slow dwell elsewhere
    t = np.arange(0, 20, 0.001)
    x, y = np.empty_like(t), np.empty_like(t)
    a, b = t < 8, (t >= 8) & (t < 14)
    x[a] = 350 + np.cumsum(rng.normal(0, 0.4, a.sum()))
    y[a] = 410 + np.cumsum(rng.normal(0, 0.4, a.sum()))
    tri = np.abs(((t[b] - 8) / 0.1) % 2 - 1)  # 10 Hz full-screen sweeps
    x[b], y[b] = 100 + 800 * tri, 100 + 600 * tri
    c = t >= 14
    x[c] = 800 + np.cumsum(rng.normal(0, 0.4, c.sum()))
    y[c] = 200 + np.cumsum(rng.normal(0, 0.4, c.sum()))
    gz = {"t": t, "x": x, "y": y, "valid": np.isfinite(x), "ppd": (ppd, ppd)}
    gz["v"] = _velocity(gz)

    m_uni = raw_map(gz, shape, scale)
    m_slow = raw_map(gz, shape, scale, weights=w_slow(gz))
    assert mass_in(m_slow, mask) > mass_in(m_uni, mask), "slow weighting should help"
    assert iou_budget(m_slow, mask) > 0.15
    assert time_to_first_hit(gz, mask, scale) < 0.1

    clip = raw_clip(gz, shape, scale, T=8)
    assert clip.shape == (8, *shape) and clip.sum() > 0

    # ideas 2/3/4 — just exercise the paths and sanity-bound the outputs
    assert w_fast(gz).sum() > 0
    assert 0 <= mass_near_boundary(m_slow, mask) <= 1
    assert w_revisit(gz, scale).sum() > 0  # sweeps re-enter cells
    assert np.allclose(w_pupil(gz), w_uniform(gz))  # no pupil -> uniform
    gz["pupil"] = np.where(t < 8, 1.5, 1.0)          # higher load during dwell
    assert w_pupil(gz).sum() > 0
    gz.pop("pupil")

    fix = pd.DataFrame({
        "x_position": [350, 340, 360, 800, 820],
        "y_position": [410, 420, 400, 200, 210],
        "timestamp_start_fixation": [0.0, 1.0, 2.0, 3.0, 4.0],
        "timestamp_end_fixation": [0.9, 1.9, 2.9, 3.2, 4.2],
        "angular_resolution_x_pixels_per_degree": ppd,
        "angular_resolution_y_pixels_per_degree": ppd,
    })
    hm = fixation_heatmap(fix, shape, scale)
    assert mass_in(hm, mask) > 0.4
    assert fixation_sequence(fix).shape == (5, 4)

    wsp = w_speech(gz, pd.DataFrame({"word": ["there", "is", "a", "nodule."],
                                     "timestamp_start_word": [1.0, 1.3, 1.5, 6.0],
                                     "timestamp_end_word": [1.2, 1.4, 1.9, 6.5]}))
    assert wsp.sum() > 0 and mass_in(raw_map(gz, shape, scale, wsp), mask) > 0.5
    print("self-check ok")


if __name__ == "__main__":
    _selfcheck()
