import numpy as np
import matplotlib.pyplot as plt


def _prepare_image(img: np.ndarray) -> np.ndarray:
    """
    Ensure the image is 2D float in [0, 1].

    Accepts 2D or 3D arrays (H, W) or (H, W, C).
    """
    arr = np.asarray(img)

    if arr.ndim == 3:
        # Convert RGB to grayscale by averaging channels
        arr = arr.mean(axis=2)

    if arr.ndim != 2:
        raise ValueError(
            f"Expected a 2D or 3D array for an image, got shape {arr.shape}"
        )

    arr = arr.astype(np.float32)
    # Normalize to [0, 1] just in case
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)

    return arr


def create_statistics_meme(
    original_img: np.ndarray,
    stipple_img: np.ndarray,
    block_letter_img: np.ndarray,
    masked_stipple_img: np.ndarray,
    output_path: str,
    dpi: int = 150,
    background_color: str = "white"
) -> None:
    """
    Assemble the four-panel statistics meme and save as a PNG.

    Parameters
    ----------
    original_img : np.ndarray
        2D array representing the prepared original image (Reality).
    stipple_img : np.ndarray
        2D array representing the stippled image (Your Model).
    block_letter_img : np.ndarray
        2D array representing the block letter mask (Selection Bias).
    masked_stipple_img : np.ndarray
        2D array representing the masked stippled image (Estimate).
    output_path : str
        Path where the final meme PNG will be saved.
    dpi : int, optional
        Dots per inch for the saved figure. Default is 150.
    background_color : str, optional
        Figure background color. Default is "white".
    """
    # Prepare / normalize all images
    orig = _prepare_image(original_img)
    stip = _prepare_image(stipple_img)
    mask_letter = _prepare_image(block_letter_img)
    masked = _prepare_image(masked_stipple_img)

    # Make sure they are all the same size by center-cropping to
    # the smallest common height/width (just in case)
    heights = [orig.shape[0], stip.shape[0], mask_letter.shape[0], masked.shape[0]]
    widths = [orig.shape[1], stip.shape[1], mask_letter.shape[1], masked.shape[1]]
    h_min = min(heights)
    w_min = min(widths)

    def center_crop(a: np.ndarray) -> np.ndarray:
        h, w = a.shape
        top = max((h - h_min) // 2, 0)
        left = max((w - w_min) // 2, 0)
        return a[top:top + h_min, left:left + w_min]

    orig_c = center_crop(orig)
    stip_c = center_crop(stip)
    letter_c = center_crop(mask_letter)
    masked_c = center_crop(masked)

    # Create figure and axes
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 4),
        constrained_layout=True
    )
    fig.patch.set_facecolor(background_color)

    panels = [
        ("Reality", orig_c),
        ("Your Model", stip_c),
        ("Selection Bias", letter_c),
        ("Estimate", masked_c),
    ]

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.axis("off")
        ax.set_facecolor(background_color)

        # Simple thin border to make it look more “panel-y”
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_edgecolor("black")

    # Optional overall title (comment out if you don't want it)
    # fig.suptitle(
    #     "Selection Bias: What You See vs. Reality",
    #     fontsize=14,
    #     fontweight="bold",
    # )

    # Save and close
    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=background_color,
        bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    # This block won't run from Quarto; it's just a dev sanity check.
    pass
