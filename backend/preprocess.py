import numpy as np

from PIL import Image, ImageOps


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Convert a user's handwritten digit image into
    an MNIST-style 28x28 grayscale image.

    Steps:
        1. Convert to grayscale
        2. Detect whether background needs inversion
        3. Find the digit bounding box
        4. Crop the digit
        5. Resize while preserving aspect ratio
        6. Center the digit on a 28x28 canvas
    """

    # ========================================================
    # 1. Convert to grayscale
    # ========================================================

    image = image.convert("L")


    # ========================================================
    # 2. Convert image to NumPy array
    # ========================================================

    image_array = np.array(image)


    # ========================================================
    # 3. Make sure digit is WHITE and background is BLACK
    # ========================================================

    #
    # MNIST roughly looks like:
    #
    # background = 0   (black)
    # digit      = 255 (white)
    #
    # If the image has a white background and dark digit,
    # invert it.
    #

    if image_array.mean() > 127:

        image = ImageOps.invert(image)

        image_array = np.array(image)


    # ========================================================
    # 4. Find pixels belonging to the digit
    # ========================================================

    # Ignore very faint pixels.

    threshold = 20

    mask = image_array > threshold


    # ========================================================
    # 5. Handle completely empty canvas
    # ========================================================

    if not np.any(mask):

        return Image.new(
            "L",
            (28, 28),
            0
        )


    # ========================================================
    # 6. Find bounding box
    # ========================================================

    rows = np.any(mask, axis=1)

    columns = np.any(mask, axis=0)


    y_indices = np.where(rows)[0]

    x_indices = np.where(columns)[0]


    top = y_indices[0]

    bottom = y_indices[-1]

    left = x_indices[0]

    right = x_indices[-1]


    # ========================================================
    # 7. Crop digit
    # ========================================================

    cropped = image.crop(
        (
            left,
            top,
            right + 1,
            bottom + 1
        )
    )


    # ========================================================
    # 8. Resize digit
    # ========================================================

    width, height = cropped.size


    # MNIST digit should fit approximately inside 20x20.

    target_size = 20


    if width > height:

        new_width = target_size

        new_height = max(
            1,
            int(
                height * target_size / width
            )
        )

    else:

        new_height = target_size

        new_width = max(
            1,
            int(
                width * target_size / height
            )
        )


    resized = cropped.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )


    # ========================================================
    # 9. Create 28x28 black canvas
    # ========================================================

    canvas = Image.new(
        "L",
        (28, 28),
        0
    )


    # ========================================================
    # 10. Center digit
    # ========================================================

    x_position = (
        28 - new_width
    ) // 2

    y_position = (
        28 - new_height
    ) // 2


    canvas.paste(
        resized,
        (
            x_position,
            y_position
        )
    )


    return canvas