"""YAMLUT library: basic utilities."""

import os
from os import PathLike
from typing import Union

import pathlib
from pathlib import Path

import cv2

from core import *

PathOrSim = Union[str, PathLike[str]]


def list_file_names(directory: PathOrSim) -> list[str]:
    files = os.listdir(str(directory))
    files = [f for f in files if os.path.isfile(directory / f)]
    return [Path(f).stem for f in files]


def load_image(file_name: PathOrSim) -> Cv2Image:
    """
    Load image from file.

    :param file_name: File name of the image.
    :return: Loaded image.
    """
    return cv2.imread(str(file_name))


def store_image(image: Cv2Image, file_name: PathOrSim) -> None:
    """
    Store image to file.

    :param image: Image to save.
    :param file_name: Path to the destination file.
    """
    cv2.imwrite(str(file_name), image)


def load_relative_rectangles(
    file_name: PathOrSim, fmt="YOLO"
) -> list[BoundingRectangle[RelativePoint2D]]:
    """
    Load relative bounding rectangles from file.

    :param file_name: Name of the file containing rectangles' data.
    :param fmt: Format in which the rectangles are saved.
    :return: List of loaded relative rectangles.
    """
    boxes = []
    with open(str(file_name), "r") as f:
        all_lines = f.readlines()

    if fmt == "YOLO":
        for line in all_lines:
            txt = line.split()
            cls_idx = int(txt[0])
            xc, yc, w, h = [float(x) for x in txt[1:5]]
            dx, dy = w / 2, h / 2
            x1, y1, x2, y2 = xc - dx, yc - dy, xc + dx, yc + dy
            boxes.append((cls_idx, (x1, y1), (x2, y2)))
        return boxes
    else:
        raise ValueError(f"Unknown contour format: {fmt}")


def load_relative_contours(
    fname: PathOrSim, fmt="YOLO"
) -> list[Contour[RelativePoint2D]]:
    """
    Load relative 2D polygonal contours from file.

    :param fname: Name of the file containing contours' data.
    :param fmt: Format in which the contours are saved.
    :return: List of loaded contours.
    """
    contours = []
    with open(str(fname), "r") as f:
        all_lines = f.readlines()

    if fmt == "YOLO":
        for line in all_lines:
            txt = line.split()
            cls_idx = int(txt[0])
            coordinates = [float(x) for x in txt[1:]]
            contours.append((cls_idx, list(zip(coordinates[0::2], coordinates[1::2]))))
        return contours
    else:
        raise ValueError(f"Unknown contour format: {fmt}")


def r2a_point2d(
    relative_point: RelativePoint2D, img_width: int, img_height: int
) -> AbsolutePoint2D:
    """
    Convert a relative point to an absolute one.

    :param relative_point: Relative point.
    :param img_width: Width of the image within which the point is contained.
    :param img_height: Height of the image within with the point is contained.
    :return: Absolute point corresponding to the given relative one.
    """
    xr, yr = relative_point
    return int(xr * img_width), int(yr * img_height)


def r2a_rectangle(
    relative_rect: BoundingRectangle[RelativePoint2D], img_width: int, img_height: int
) -> BoundingRectangle[AbsolutePoint2D]:
    """
    Convert a relative rectangle to an absolute one.

    :param relative_rect: Relative rectangle.
    :param img_width: Width of the image within which the point is contained.
    :param img_height: Height of the image within with the point is contained.
    :return: Absolute rectangle corresponding to the given relative one.
    """
    r2a = lambda p: r2a_point2d(p, img_width, img_height)
    ndx, p1, p2 = relative_rect
    return ndx, r2a(p1), r2a(p2)


def r2a_contour(
    relative_contour: Contour[RelativePoint2D], img_width: int, img_height: int
) -> Contour[AbsolutePoint2D]:
    """
    Convert a relative contour to an absolute one.

    :param relative_contour: Relative contour.
    :param img_width: Width of the image within which the contour is contained.
    :param img_height: Height of the image within with the contour is contained.
    :return: Absolute contour corresponding to the given relative one.
    """
    r2a = lambda p: r2a_point2d(p, img_width, img_height)
    cls_index, relative_points = relative_contour
    return cls_index, [r2a(p) for p in relative_points]


def resize_image(image: Cv2Image, height: int = None, width: int = None) -> Cv2Image:
    """
    Resize image to the given height and width.

    If height or width are not given, the image will be resized in a manner that
    preserves the aspect ratio.

    If neither height nor width are given, a copy of the original image will be returned.

    :param image: The image to resize.
    :param height: Image height after resize (or `None`, which is default).
    :param width: Image width after resize (or `None`, which is default.)
    :return: The resized image.
    """
    if height is None and width is None:
        return image.copy()

    h, w = image.shape[:2]
    # The case of unspecified height
    if height is None:
        scale = width / w
        height = int(h * scale)
        return cv2.resize(image, (width, height))
    # The case of unspecified width
    if width is None:
        scale = height / h
        width = int(w * scale)
        return cv2.resize(image, (width, height))
    # The case when both dimensions are specified
    return cv2.resize(image, (width, height))


def create_mask_from_rectangles(
    rectangles: list[BoundingRectangle[AbsolutePoint2D]],
    image_width: int,
    image_height: int,
) -> Cv2BWImage:
    mask = np.zeros(shape=(image_height, image_width), dtype=np.uint8)
    for _, (x_ul, y_ul), (x_lr, y_lr) in rectangles:
        x_ul, y_ul = max(0, x_ul), max(0, y_ul)
        x_lr, y_lr = min(image_width, x_lr), min(image_height, y_lr)
        mask[y_ul:y_lr, x_ul:x_lr] = 255
    return mask


def create_mask_from_contours(
    contours: list[Contour[AbsolutePoint2D]], image_width: int, image_height: int
) -> Cv2BWImage:
    from visualization import draw_contour

    mask = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)
    for c in contours:
        mask = draw_contour(
            mask, c, color=(255, 255, 255), fill=(255, 255, 255), width=1
        )
    return mask[:, :, 1]
