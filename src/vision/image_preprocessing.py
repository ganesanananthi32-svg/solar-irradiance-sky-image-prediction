"""Sky Image Preprocessing and Cloud Detection.

Handles circular fisheye masking, solar glare detection, Red-to-Blue Ratio (RBR)
cloud segmentation, and whole-sky feature extraction.
"""

import numpy as np
from PIL import Image
from typing import Tuple, Optional, Union


class SkyImagePreprocessor:
    """Preprocesses all-sky images for deep learning and physics feature extraction."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 128),
        mask_radius_ratio: float = 0.95,
        rbr_threshold: float = 0.75
    ):
        """
        Args:
            target_size: (height, width) to resize images to.
            mask_radius_ratio: Ratio of half the image dimension for circular mask.
            rbr_threshold: Red-to-Blue ratio threshold separating cloud from clear sky.
        """
        self.target_size = target_size
        self.mask_radius_ratio = mask_radius_ratio
        self.rbr_threshold = rbr_threshold

    def create_circular_mask(self, h: int, w: int) -> np.ndarray:
        """Creates a boolean mask True inside the circular sky dome and False outside."""
        cy, cx = h / 2.0, w / 2.0
        radius = min(h, w) / 2.0 * self.mask_radius_ratio
        y_indices, x_indices = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
        return dist_from_center <= radius

    def preprocess_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loads and preprocessed a sky image.

        Returns:
            rgb_norm: Normalized (H, W, 3) float32 array in [0, 1] with blacked-out perimeter.
            sky_mask: Boolean (H, W) mask of valid sky pixels.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype == np.uint8:
                img = Image.fromarray(image_input)
            else:
                img = Image.fromarray((np.clip(image_input, 0, 1) * 255).astype(np.uint8))
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Resize to standard model input dimensions
        img_resized = img.resize(self.target_size, Image.Resampling.BILINEAR)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0

        # Mask circular sky dome
        h, w = self.target_size
        mask = self.create_circular_mask(h, w)
        img_arr[~mask] = 0.0

        return img_arr, mask

    def compute_rbr_cloud_mask(
        self,
        rgb_image: np.ndarray,
        sky_mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """Computes Red-to-Blue Ratio (RBR = R / B) and cloud coverage fraction.

        Clear sky Rayleigh scattering: R is low, B is high -> RBR < 0.6.
        Cloud Mie scattering: R and B are similar -> RBR > 0.75 - 0.90.

        Args:
            rgb_image: (H, W, 3) array normalized in [0, 1].
            sky_mask: Boolean mask of valid sky pixels.

        Returns:
            cloud_mask: Boolean (H, W) mask where True indicates cloud.
            cloud_cover_fraction: Float between 0.0 (clear) and 1.0 (overcast).
        """
        if sky_mask is None:
            sky_mask = self.create_circular_mask(rgb_image.shape[0], rgb_image.shape[1])

        r = rgb_image[:, :, 0]
        b = rgb_image[:, :, 2]

        epsilon = 1e-4
        rbr = r / (b + epsilon)

        # Clouds are pixels where RBR exceeds threshold
        cloud_pixels = (rbr >= self.rbr_threshold) & sky_mask

        total_sky_pixels = np.count_nonzero(sky_mask)
        cloud_count = np.count_nonzero(cloud_pixels)

        cloud_cover_fraction = float(cloud_count / total_sky_pixels) if total_sky_pixels > 0 else 0.0

        return cloud_pixels, cloud_cover_fraction

    def extract_clear_sky_background_rbr(
        self,
        rgb_image: np.ndarray,
        sky_mask: np.ndarray
    ) -> float:
        """Extracts average RBR of the clearest sky region for atmospheric haze calibration."""
        r = rgb_image[:, :, 0]
        b = rgb_image[:, :, 2]
        rbr = r / (b + 1e-4)

        valid_rbr = rbr[sky_mask]
        if len(valid_rbr) == 0:
            return 0.45

        # Take 15th percentile of RBR (represents clearest background sky)
        clear_rbr = float(np.percentile(valid_rbr, 15))
        return clear_rbr
