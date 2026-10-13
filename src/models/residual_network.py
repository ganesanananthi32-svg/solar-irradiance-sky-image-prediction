"""Gao & Liu (WACV 2022) Benchmark Model Implementation.

Replicates the Clear-Sky Residual Learning Framework from:
"Short-Term Solar Irradiance Prediction From Sky Images With a Clear Sky Model"
(Huiyu Gao & Miaomiao Liu, WACV 2022).

Architecture:
1. Spatial-temporal feature extraction from sequence of sky images.
2. Fusion with auxiliary physical/meteorological features (zenith, azimuth, temp, humidity).
3. Prediction of the Solar Irradiance Residual: Delta GHI = GHI - GHI_cs.
4. Final forecast reconstructs GHI = GHI_cs + Delta GHI.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple


class GaoLiuResidualNetwork:
    """Clear-Sky Residual Network replicating Gao & Liu (WACV 2022)."""

    def __init__(
        self,
        image_feature_dim: int = 64,
        aux_dim: int = 5,
        hidden_dim: int = 128,
        seed: int = 42
    ):
        """
        Args:
            image_feature_dim: Dimension of extracted visual embeddings.
            aux_dim: Number of auxiliary features (zenith, azimuth, air temp, humidity, wind).
            hidden_dim: Intermediate hidden dimension.
        """
        np.random.seed(seed)
        self.image_feature_dim = image_feature_dim
        self.aux_dim = aux_dim
        self.in_dim = image_feature_dim + aux_dim
        self.hidden_dim = hidden_dim

        # Layer 1: Fusion
        scale1 = np.sqrt(2.0 / self.in_dim)
        self.W1 = np.random.randn(self.in_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)

        # Layer 2: Representation refinement
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim // 2) * scale2
        self.b2 = np.zeros(hidden_dim // 2)

        # Layer 3: Residual Head (outputs Delta GHI)
        scale3 = np.sqrt(2.0 / (hidden_dim // 2))
        self.W3 = np.random.randn(hidden_dim // 2, 1) * scale3
        self.b3 = np.zeros(1)

        # Adam optimizer state
        self.m = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]]
        self.v = [np.zeros_like(p) for p in [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]]
        self.t = 0

    def extract_image_features(self, images: np.ndarray) -> np.ndarray:
        """Extracts spatial-temporal cloud features from a batch of sky image frames.

        Args:
            images: (Batch, Seq_len, H, W, Channels) or (Batch, H, W, C).
        """
        if images.ndim == 4:
            # Single frame per sample (B, H, W, C)
            images = images[:, np.newaxis, ...]

        b, seq, h, w, c = images.shape
        features = np.zeros((b, self.image_feature_dim), dtype=np.float32)

        for i in range(b):
            sample = images[i]
            # Spatial statistics across channels
            r_mean = np.mean(sample[..., 0])
            g_mean = np.mean(sample[..., 1])
            b_mean = np.mean(sample[..., 2])
            r_std = np.std(sample[..., 0])
            b_std = np.std(sample[..., 2])
            rbr = r_mean / (b_mean + 1e-4)

            # Center patch (circumsolar approximation) vs perimeter variance
            ch, cw = h // 2, w // 2
            center_patch = sample[:, ch - h//4 : ch + h//4, cw - w//4 : cw + w//4, :]
            center_mean = np.mean(center_patch)
            center_var = np.var(center_patch)

            # Temporal difference across frames
            if seq > 1:
                frame_diff = np.mean(np.abs(sample[-1] - sample[0]))
                frame_ramp = np.mean(sample[-1]) - np.mean(sample[0])
            else:
                frame_diff = 0.0
                frame_ramp = 0.0

            # Deterministic projection vector to feature_dim
            stats = np.array([
                r_mean, g_mean, b_mean, r_std, b_std, rbr,
                center_mean, center_var, frame_diff, frame_ramp
            ], dtype=np.float32)

            # Repeat and tile to match self.image_feature_dim
            tile_count = int(np.ceil(self.image_feature_dim / len(stats)))
            tiled = np.tile(stats, tile_count)[:self.image_feature_dim]
            features[i] = tiled

        return features

    def predict_residual(
        self,
        image_features: np.ndarray,
        aux_features: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Predicts Delta GHI = GHI - GHI_cs."""
        x = np.concatenate([image_features, aux_features], axis=-1)

        # Forward pass
        h1 = np.maximum(0.0, np.dot(x, self.W1) + self.b1) # ReLU
        h2 = np.maximum(0.0, np.dot(h1, self.W2) + self.b2) # ReLU
        residual = np.dot(h2, self.W3) + self.b3

        cache = {"x": x, "h1": h1, "h2": h2, "pred_residual": residual}
        return residual.squeeze(-1), cache

    def predict(
        self,
        images: np.ndarray,
        aux_features: np.ndarray,
        ghi_cs: np.ndarray
    ) -> np.ndarray:
        """Computes final GHI forecast: GHI = GHI_cs + predicted_residual.

        Enforces physical non-negativity constraint.
        """
        img_feats = self.extract_image_features(images)
        pred_res, _ = self.predict_residual(img_feats, aux_features)
        final_ghi = ghi_cs + pred_res
        return np.maximum(0.0, final_ghi)

    def train_step(
        self,
        images: np.ndarray,
        aux_features: np.ndarray,
        ghi_cs: np.ndarray,
        ghi_target: np.ndarray,
        lr: float = 0.001
    ) -> float:
        """Performs one Adam gradient descent step on MSE(GHI_cs + Delta GHI, GHI_target)."""
        target_residual = (ghi_target - ghi_cs)[:, np.newaxis]
        img_feats = self.extract_image_features(images)
        pred_res, cache = self.predict_residual(img_feats, aux_features)
        pred_res_2d = pred_res[:, np.newaxis]

        # MSE Loss gradient on residual
        n = len(images)
        d_out = 2.0 * (pred_res_2d - target_residual) / n

        # Backward pass
        dW3 = np.dot(cache["h2"].T, d_out)
        db3 = np.sum(d_out, axis=0)

        dh2 = np.dot(d_out, self.W3.T) * (cache["h2"] > 0)
        dW2 = np.dot(cache["h1"].T, dh2)
        db2 = np.sum(dh2, axis=0)

        dh1 = np.dot(dh2, self.W2.T) * (cache["h1"] > 0)
        dW1 = np.dot(cache["x"].T, dh1)
        db1 = np.sum(dh1, axis=0)

        # Adam optimization
        params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        grads = [dW1, db1, dW2, db2, dW3, db3]
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g ** 2)
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)

        loss = float(np.mean((pred_res_2d.squeeze() - target_residual.squeeze()) ** 2))
        return loss
