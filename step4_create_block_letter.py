import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional


def _get_font(font_size: int) -> ImageFont.FreeTypeFont:
    """
    Try to load a bold-ish font that should exist on most systems.
    Fall back to the default PIL font if none are found.
    """
    font_candidates = [
        # Common cross-platform font names
        "DejaVuSans-Bold.ttf",
        "Arial Bold.ttf",
        "Arialbd.ttf",
        "Arial.ttf",
        # Windows absolute paths (in case relative lookup fails)
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]

    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception:
            continue

    # Fallback: default bitmap font (not ideal but better than crashing)
    return ImageFont.load_default()


def create_block_letter_s(
    height: int,
    width: int,
    letter: str = "S",
    font_size_ratio: float = 0.9
) -> np.ndarray:
    """
    Create a block letter mask matching the given image dimensions.

    Parameters
    ----------
    height : int
        Height of the output image in pixels.
    width : int
        Width of the output image in pixels.
    letter : str, optional
        Letter to draw (default is "S").
    font_size_ratio : float, optional
        Fraction of the smaller dimension used as the font size
        (0 < font_size_ratio <= 1). Default is 0.9.

    Returns
    -------
    np.ndarray
        2D array of shape (height, width) with values in [0, 1],
        where 0.0 = black letter and 1.0 = white background.
    """
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive integers")

    # Clamp font_size_ratio to (0, 1]
    font_size_ratio = max(0.1, min(1.0, font_size_ratio))

    # Create a white background grayscale image
    img = Image.new("L", (width, height), color=255)  # "L" = 8-bit grayscale
    draw = ImageDraw.Draw(img)

    # Choose font size based on the smaller dimension
    base_size = int(min(height, width) * font_size_ratio)
    font = _get_font(base_size)

    # Compute bounding box for the letter to center it
    text = str(letter)[0] if letter else "S"

    # Newer Pillow has textbbox; fall back to textsize if unavailable
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = draw.textsize(text, font=font)

    # Center the letter
    x = (width - text_width) / 2
    y = (height - text_height) / 2

    # Draw the letter in black on white background
    draw.text((x, y), text, font=font, fill=0)

    # Convert to numpy array in [0, 1]
    arr = np.array(img).astype(np.float32) / 255.0

    return arr


if __name__ == "__main__":
    # Simple manual test (optional): saves a preview image
    h, w = 512, 512
    mask = create_block_letter_s(h, w, letter="S", font_size_ratio=0.9)
    preview = Image.fromarray((mask * 255).astype(np.uint8))
    preview.save("debug_block_letter_s.png")
