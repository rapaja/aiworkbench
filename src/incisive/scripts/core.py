"""YAMLUT library: core types and definitions."""

from typing import TypeVar, Union, Tuple, List
import numpy as np


Color = tuple[int, int, int]
ClassIndex = int
ClassLabel = str

Cv2BWImage = np.ndarray
Cv2GrayImage = np.ndarray
Cv2ColorImage = np.ndarray
Cv2Image = Union[Cv2BWImage, Cv2GrayImage, Cv2ColorImage]

PointType = TypeVar("PointType")
Contour = Tuple[ClassIndex, list[PointType]]
BoundingRectangle = Tuple[ClassIndex, PointType, PointType]

RelativePoint2D = Tuple[float, float]
AbsolutePoint2D = Tuple[int, int]
