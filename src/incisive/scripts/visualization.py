"""YAMLUT library: visualization utilities."""

import cv2
from PIL import Image, ImageDraw

from core import *


def draw_rectangle(
    image: Cv2Image,
    rectangle: BoundingRectangle[AbsolutePoint2D],
    color: Color = (255, 0, 0),
    width: int = 5,
) -> Cv2Image:
    """
    Draw rectangle on the given image canvas.

    :param image: Image to draw on.
    :param rectangle: Rectangle to draw.
    :param color: Outline color of the rectangle.
    :param width: Outline width.
    :return: Annotated image.
    """
    pil_image = Image.fromarray(image)
    canvas = ImageDraw.Draw(pil_image)
    _, p1, p2 = rectangle
    canvas.rectangle((p1, p2), outline=color, width=width)
    return np.asarray(pil_image).astype(np.uint8)


def draw_contour(
    image: Cv2Image,
    contour: Contour[AbsolutePoint2D],
    color: Color = (255, 0, 0),
    fill: Color = None,
    width: int = 5,
) -> Cv2Image:
    """
    Draw contour on the given image.

    :param fill:
    :param image: Image canvas to draw on.
    :param contour: Contour to draw.
    :param color: Outline color of the contour.
    :param width: Outline width.
    :return: Annotated image.
    """
    pil_image = Image.fromarray(image)
    canvas = ImageDraw.Draw(pil_image)
    _, polygon = contour
    canvas.polygon(polygon, outline=color, fill=fill, width=width)
    return np.asarray(pil_image).astype(np.uint8)


def monochrome_image(
    color: Color, image_width: int, image_height: int
) -> Cv2ColorImage:
    """
    Create a monochrome image of the given shape and color.

    :param color: Color of the created image.
    :param image_width: Created image width.
    :param image_height: Created image height.
    :return: Monochrome image of the given shape.
    """
    uim = np.ones(shape=(image_height, image_width))
    r, g, b = color
    result = cv2.merge((b * uim, g * uim, r * uim))
    return result.astype(np.uint8)


def concat_h(images: list[Cv2ColorImage]) -> Cv2ColorImage:
    if not images:
        return None

    total_width = sum([im.shape[1] for im in images])
    height = images[0].shape[0]

    assert all(
        [[im.shape[0] == height for im in images]]
    ), "All images must be of the same height"

    new_image = np.zeros(shape=(height, total_width, 3), dtype=np.uint8)
    w = 0
    for i in range(len(images)):
        width = images[i].shape[1]
        new_image[:, w : w + width, :] = images[i]
        w += width

    return new_image


def blend_rect(
    image: Cv2Image,
    rectangles: list[BoundingRectangle[AbsolutePoint2D]],
    fill: Color,
    alpha=0.5,
):
    import utils

    mask = utils.create_mask_from_rectangles(rectangles, image.shape[1], image.shape[0])
    overlay = cv2.merge([mask.copy(), mask.copy(), mask.copy()])
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if mask[r, c] == 255:
                overlay[r, c, :] = fill
            else:
                overlay[r, c, :] = image[r, c, :]
    overlay = alpha * image + (1 - alpha) * overlay
    overlay = overlay.astype(np.uint8)
    return overlay


def blend_contours(
    image: Cv2Image,
    contours: list[Contour[AbsolutePoint2D]],
    fill: Color,
    alpha=0.5,
):
    import utils

    mask = utils.create_mask_from_contours(contours, image.shape[1], image.shape[0])
    overlay = cv2.merge([mask.copy(), mask.copy(), mask.copy()])
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            if mask[r, c] == 255:
                overlay[r, c, :] = fill
            else:
                overlay[r, c, :] = image[r, c, :]
    overlay = alpha * image + (1 - alpha) * overlay
    overlay = overlay.astype(np.uint8)
    return overlay
