import numpy as np


def create_masked_stipple(
    stipple_img: np.ndarray,
    mask_img: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Apply a block-letter mask to a stippled image to simulate selection bias.

    Parameters
    ----------
    stipple_img : np.ndarray
        2D array (H x W) with values in [0, 1], where 0 = black dots,
        1 = white background (your stippled image).
    mask_img : np.ndarray
        2D array (H x W) with values in [0, 1], where 0 = black mask area
        and 1 = white background (your block letter image).
    threshold : float, optional
        Pixels in the mask with values < threshold are considered part of
        the "mask" region and will have stipples removed (set to white/1.0).
        Default is 0.5.

    Returns
    -------
    np.ndarray
        2D array (H x W) with values in [0, 1], where pixels in the masked
        region are white (1.0) and other pixels keep the original stipple
        values.
    """
    # Basic shape check
    if stipple_img.shape != mask_img.shape:
        raise ValueError(
            f"stipple_img and mask_img must have the same shape, "
            f"got {stipple_img.shape} and {mask_img.shape}"
        )

    # Ensure float32 and clamp to [0, 1]
    stipple = np.clip(stipple_img.astype(np.float32), 0.0, 1.0)
    mask = np.clip(mask_img.astype(np.float32), 0.0, 1.0)

    # True where we want to remove data (inside the letter / dark region)
    mask_region = mask < threshold

    # Where mask_region is True → set to 1.0 (white), else keep stipple value
    result = np.where(mask_region, 1.0, stipple)

    return result


if __name__ == "__main__":
    # Optional quick self-test (won't run in your Quarto doc)
    h, w = 4, 4
    stipple_test = np.zeros((h, w), dtype=np.float32)  # all dots (black)
    mask_test = np.ones((h, w), dtype=np.float32)      # all white
    mask_test[1:3, 1:3] = 0.0                          # dark square in center

    masked = create_masked_stipple(stipple_test, mask_test, threshold=0.5)
    print("Stipple:\n", stipple_test)
    print("Mask:\n", mask_test)
    print("Masked result:\n", masked)
