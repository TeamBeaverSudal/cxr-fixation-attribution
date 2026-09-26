"""Finding-conditioned fixation selector used in the reference study."""

import numpy as np

from finding_level_gaze_targets.maps.core import WORD_DIM, align_feats, inside_ellipses


FOURIER_LEVELS = 4
POSITION_DIM = 4 * FOURIER_LEVELS


def fourier_feat(coordinates, levels=FOURIER_LEVELS):
    """Encode normalized two-dimensional fixation coordinates."""
    frequencies = (2.0 ** np.arange(levels)).astype(np.float32) * np.pi
    angles = coordinates[:, :, None].astype(np.float32) * frequencies[None, None, :]
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=-1).reshape(
        len(coordinates), -1
    ).astype(np.float32)


def pos_feat(coordinates, mode):
    if mode == "raw":
        return coordinates.astype(np.float32)
    if mode != "fourier":
        raise ValueError(f"unknown positional encoding: {mode}")
    return fourier_feat(coordinates)


def _position_dim(use_position, mode):
    if not use_position:
        return 0
    return 2 if mode == "raw" else POSITION_DIM


def make_net(labels, use_position, use_text=False, fusion="concat", pos_mode="fourier"):
    """Construct the selector architecture for a specified feature set."""
    import torch.nn as nn

    from finding_level_gaze_targets.maps.core import temporal_dim

    position_dim = _position_dim(use_position, pos_mode)
    text_dim = WORD_DIM if use_text else 0

    if fusion == "crossattn":
        key_width = 16
        key_input_dim = temporal_dim() + 8 + position_dim

        class CrossAttentionSelector(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(len(labels), 8)
                self.key_projection = nn.Linear(key_input_dim, key_width)
                self.query_projection = nn.Linear(8 + text_dim, key_width)

            def attn(self, features, label, position, word_features=None):
                import torch

                finding = self.embedding(torch.tensor(label))
                key_parts = [features, finding.expand(len(features), -1)]
                if use_position:
                    key_parts.append(torch.from_numpy(position))
                keys = self.key_projection(torch.cat(key_parts, dim=1))
                query_parts = [finding]
                if use_text:
                    query_parts.append(torch.from_numpy(word_features))
                query = self.query_projection(torch.cat(query_parts, dim=0))
                scores = (keys @ query) / (key_width**0.5)
                return torch.softmax(scores, dim=0)

        return CrossAttentionSelector()

    if fusion != "concat":
        raise ValueError(f"unknown fusion architecture: {fusion}")
    input_dim = temporal_dim() + 8 + position_dim + text_dim

    class ConcatenationSelector(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(len(labels), 8)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def attn(self, features, label, position, word_features=None):
            import torch

            finding = self.embedding(torch.tensor(label)).expand(len(features), -1)
            parts = [features, finding]
            if use_position:
                parts.append(torch.from_numpy(position))
            if use_text:
                parts.append(torch.from_numpy(word_features).expand(len(features), -1))
            scores = self.mlp(torch.cat(parts, dim=1)).squeeze(-1)
            return torch.softmax(scores, dim=0)

    return ConcatenationSelector()


def train_model(
    training_items,
    labels,
    use_position,
    epochs,
    use_text=False,
    fusion="concat",
    pos_mode="fourier",
    seed=0,
):
    """Fit one selector with a deterministic optimizer seed and data order."""
    import os
    import torch

    torch.set_num_threads(max(1, os.cpu_count() - 1))
    torch.manual_seed(seed)
    np.random.seed(seed)
    items = list(training_items)
    network = make_net(labels, use_position, use_text, fusion, pos_mode)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
    empty_position = np.zeros((0, _position_dim(use_position, pos_mode)), np.float32)

    for epoch in range(epochs):
        network.train()
        np.random.shuffle(items)
        total_mass = 0.0
        for item in items:
            if use_text:
                fixations, mentions, ellipses, label, word_features = item[:5]
            else:
                fixations, mentions, ellipses, label = item[:4]
                word_features = None
            inside = torch.from_numpy(
                inside_ellipses(fixations, ellipses).astype(np.float32)
            )
            if inside.sum() == 0:
                continue
            position = (
                pos_feat(fixations[:, :2], pos_mode)
                if use_position
                else empty_position
            )
            weights = network.attn(
                torch.from_numpy(align_feats(fixations, mentions)),
                label,
                position,
                word_features,
            )
            mass_inside = (weights * inside).sum()
            entropy = -(weights * (weights + 1e-9).log()).sum()
            loss = -torch.log(mass_inside + 1e-6) - 0.01 * entropy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_mass += mass_inside.item()
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"    epoch {epoch} train mass-in "
                f"{total_mass / max(len(items), 1):.4f}",
                flush=True,
            )
    network.eval()
    return network


def predict_raw(
    network,
    fixations,
    mentions,
    label,
    use_position,
    shuffle_pos=False,
    rng=None,
    use_text=False,
    wf=None,
    pos_mode="fourier",
    return_weights=False,
):
    """Return fixation weights or their unblurred rendering."""
    import torch

    selected = fixations
    if shuffle_pos:
        if rng is None:
            raise ValueError("rng is required when shuffle_pos=True")
        selected = fixations.copy()
        permutation = rng.permutation(len(selected))
        selected[:, :2] = fixations[permutation, :2]
    position = (
        pos_feat(selected[:, :2], pos_mode)
        if use_position
        else np.zeros((len(selected), 0), np.float32)
    )
    with torch.no_grad():
        weights = network.attn(
            torch.from_numpy(align_feats(selected, mentions)),
            label,
            position,
            wf,
        ).numpy()
    if return_weights:
        return weights
    return _raw_grid(selected[:, 0], selected[:, 1], weights)


def _raw_grid(x_coordinates, y_coordinates, weights):
    from finding_level_gaze_targets.maps.core import HEAT_RES

    heatmap = np.zeros((HEAT_RES, HEAT_RES), np.float32)
    x_indices = np.clip((x_coordinates * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    y_indices = np.clip((y_coordinates * HEAT_RES).astype(int), 0, HEAT_RES - 1)
    np.add.at(heatmap, (y_indices, x_indices), weights)
    return heatmap


def blur_norm(raw_map, sigma):
    from scipy.ndimage import gaussian_filter

    heatmap = gaussian_filter(raw_map, sigma)
    maximum = heatmap.max()
    return heatmap / maximum if maximum > 0 else heatmap
