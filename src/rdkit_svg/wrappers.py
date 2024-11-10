"""Wrapper for SVG elements."""

from __future__ import annotations

import itertools
import re
from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

from typing_extensions import Self, override

from rdkit_svg.path_bbox import BBox, get_segments_with_bbox

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rdkit_svg.path_bbox import BBox


def filtered_tag(tag: str) -> str:
    """Get the tag without the provider.

    Examples:
        >>> filtered_tag("{http://www.w3.org/2000/svg}circle")
        'circle'
        >>> filtered_tag("circle")
        'circle'
    """
    return re.sub(r"\{.*\}", "", tag)


def filtered_split(s: str, sep: str | re.Pattern[str], maxsplit: int = 0) -> list[str]:
    r"""Split a string and remove empty strings.

    Examples:
        >>> filtered_split("a1b1c1", "1")
        ['a', 'b', 'c']
        >>> filtered_split("a12b12c", re.compile(r"\d+"))
        ['a', 'b']
    """
    if isinstance(sep, str):
        sep = re.compile(sep)

    return [x.strip() for x in sep.split(s, maxsplit) if x.strip()]


class Point(complex):
    """A point in 2D space. Wrapper for complex numbers."""

    @override
    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"

    @property
    def x(self) -> float:
        """Rounded x coordinate."""
        return round(self.real, 5)

    @property
    def y(self) -> float:
        """Rounded y coordinate."""
        return round(self.imag, 5)

    @property
    def text(self) -> str:
        """The point as space-separated text."""
        return f"{self.x} {self.y}"

    @override
    def __mul__(self, other: complex) -> Self:
        return self.__class__(super().__mul__(other))

    @override
    def __add__(self, other: complex) -> Self:
        return self.__class__(super().__add__(other))

    @override
    def __sub__(self, other: complex) -> Self:
        return self.__class__(super().__sub__(other))

    @override
    def __truediv__(self, other: complex) -> Self:
        return self.__class__(super().__truediv__(other))

    @override
    def __neg__(self) -> Self:
        return self.__class__(super().__neg__())

    @classmethod
    def min(cls, *points: Point) -> Point:
        """Get the minimum point."""
        return cls(min(x.real for x in points), min(x.imag for x in points))

    @classmethod
    def max(cls, *points: Point) -> Point:
        """Get the maximum point."""
        return cls(max(x.real for x in points), max(x.imag for x in points))


class Box:
    """A box defined by two points (upper left and lower right)."""

    def __init__(self, upper_left: Point, lower_right: Point) -> None:
        """Initialize the box.

        Args:
            upper_left: The upper left point of the box.
            lower_right: The lower right point of the box.
        """
        self.upper_left = upper_left
        self.lower_right = lower_right

    @override
    def __repr__(self) -> str:
        return f"Box({self.upper_left}, {self.lower_right})"

    def __iter__(self) -> Iterator[Point]:
        yield self.upper_left
        yield self.lower_right

    def _contains(self, point: Point) -> bool:
        """Check if a point is contained in the box."""
        return (
            self.upper_left.x <= point.x <= self.lower_right.x
            and self.upper_left.y <= point.y <= self.lower_right.y
        )

    def __contains__(self, key: Point | Box) -> bool:
        """Checks if a single points or a box is contained in the box."""
        if isinstance(key, Point):
            return self._contains(key)

        return all(self._contains(p) for p in key)


FILL_DEFAULTS = {
    k: "black" for k in ("circle", "ellipse", "path", "polygon", "rect", "text")
}


class ElemSpan(ABC):
    """Abstract base class for SVG elements."""

    def __init__(self, elem: ET.Element, tree: ET.Element) -> None:
        """Initialize the element.

        Args:
            elem: The element to initialize.
            tree: The tree to initialize.
        """
        self.elem = elem
        self.attr = elem.attrib
        self.tree = tree
        self.fix_style()

    def fix_style(self) -> None:
        """Split the style attribute into separate attributes."""
        if "style" not in self.attr:
            return

        style = self.attr["style"]
        styles = [x.strip() for x in style.split(";") if x.strip()]
        items = [x.split(":") for x in styles]
        for key, value in items:
            self.attr[key.strip()] = value.strip()
        del self.attr["style"]

    @override
    def __repr__(self) -> str:
        cls = self.attr.get("class", "")
        cls_suffix = f" ({cls})" if cls else ""
        return f"{filtered_tag(self.tag)}{cls_suffix}"

    @property
    def special_border(self) -> bool:
        """If the element has a special border."""
        return False

    @property
    def tag(self) -> str:
        """The tag of the element."""
        return self.elem.tag

    @property
    def has_border(self) -> bool:
        """If the element has a border."""
        stroke = self.attr.get("stroke", "none")
        stroke_width = self.attr.get("stroke-width", 1)
        return stroke != "none" and stroke_width != 0

    @property
    def hidden(self) -> bool:
        """If the element is hidden."""
        fill_def = FILL_DEFAULTS.get(self.tag, "none")
        fill = self.attr.get("fill", fill_def)
        no_fill = fill in {"none", "transparent"}
        no_border = not self.has_border
        is_transparent = self.attr.get("opacity", 1) == 0
        not_displayed = self.attr.get("display", "inline") == "none"
        return not_displayed or (no_border and no_fill) or is_transparent

    def hide(self) -> Self:
        """Set element to hidden. Modfies in place."""
        self.set_attrs(display="none")
        return self

    def get_bbox(self) -> BBox:
        """Get the bounding box of the element or None if failed."""
        raise NotImplementedError(
            f"Method get_bbox not implemented for {ET.tostring(self.elem).decode('utf-8')}"  # noqa: E501
        )

    def get_bounds(self) -> tuple[Point | None, Point | None]:
        """Get the bounds of the element."""
        bbox = self.get_bbox()

        padding = self.stroke_padding

        x, y, w, h = bbox
        return (Point(x, y) - padding, Point(x + w, y + h) + padding)

    @abstractmethod
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        """Adjust the element to the new bounds."""
        return self

    def as_float(self, *args: str) -> tuple[float, ...]:
        """Get the attributes as floats without the 'px' suffix."""
        return tuple(float(self.attr[arg].removesuffix("px")) for arg in args)

    def set_attrs(self, **kwargs: float | int | str) -> None:  # noqa: PYI041
        """Set the attributes of the element.

        Floats are rounded to 5 decimal places.
        """
        for key, raw_value in kwargs.items():
            # convert to int if possible
            if isinstance(raw_value, float):
                value = str(round(raw_value, 5))
            else:
                value = str(raw_value)
            self.attr[key] = value

    @property
    def stroke_padding(self) -> Point:
        """The padding for the stroke."""
        if self.attr.get("stroke", "none") == "none":
            stroke_padding = 0.0
        else:
            stroke_padding = (
                float(self.attr.get("stroke-width", "1").removesuffix("px")) / 2
            )

        return Point(stroke_padding, stroke_padding)


class Svg(ElemSpan):
    """SVG class."""

    @override
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        new_size = lower_right - upper_left
        self.set_attrs(width="100%", height="100%", viewBox=f"0 0 {new_size.text}")
        return self


class Text(ElemSpan):
    """Text class."""

    def _get_values(self, key: str) -> list[float]:
        values = filtered_split(self.attr[key], r"[,\s]")
        return [float(x.removesuffix("px")) for x in values]

    @override
    def get_bbox(self) -> tuple[float, float, float, float]:
        x_values = self._get_values("x")
        y_values = self._get_values("y")

        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        font_size = float(self.attr["font-size"].removesuffix("px"))
        y_min -= font_size

        max_width = 0.0
        for pair in itertools.pairwise(x_values):
            max_width = max(max_width, pair[1] - pair[0])
        x_max += max_width

        return (x_min, y_min, x_max - x_min, y_max - y_min)

    @override
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        del lower_right  # only need upper left

        def _adjust_coords(key: str, value: float) -> str:
            values = self._get_values(key)
            new_values = [x - value for x in values]
            return " ".join(str(x) for x in new_values)

        self.attr["x"] = _adjust_coords("x", upper_left.x)
        self.attr["y"] = _adjust_coords("y", upper_left.y)
        return self


class Marker(ElemSpan):
    """Marker class."""

    @override
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        new_size = lower_right - upper_left
        self.set_attrs(width="100%", height="100%", viewBox=f"0 0 {new_size.text}")
        return self


class Rect(ElemSpan):
    """Rectangle class."""

    @override
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        del lower_right  # only need upper left
        x, y = self.as_float("x", "y")
        new_lu = Point(x, y) - upper_left
        self.set_attrs(x=new_lu.x, y=new_lu.y)
        return self


class Polygon(ElemSpan):
    """Polygon class."""

    @override
    @property
    def special_border(self) -> bool:
        return True

    def _get_points(self) -> list[Point]:
        # split on comma or space
        points_as_str = filtered_split(self.attr["points"], r"[,\s]")
        points_as_floats = [float(x.removesuffix("px")) for x in points_as_str]
        x__, y__ = points_as_floats[::2], points_as_floats[1::2]
        return [Point(*x) for x in zip(x__, y__, strict=False)]

    @override
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        del lower_right  # only need upper left
        points = self._get_points()
        new_points = [p - upper_left for p in points]
        new_points_str = " ".join([p.text for p in new_points])
        self.set_attrs(points=new_points_str)
        return self


class Ellipse(ElemSpan):
    """Ellipse class."""

    @override
    def get_bbox(self) -> BBox:
        cx, cy, rx, ry = self.as_float("cx", "cy", "rx", "ry")
        return (cx - rx, cy - ry, 2 * rx, 2 * ry)

    @override
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        cx, cy = self.as_float("cx", "cy")
        new_lu = Point(cx, cy) - upper_left
        self.set_attrs(cx=new_lu.x, cy=new_lu.y)
        return self


class Circle(Ellipse):
    """Circle class."""


class Line(ElemSpan):
    """Line class."""

    @override
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        x1, y1, x2, y2 = self.as_float("x1", "y1", "x2", "y2")
        new_start = Point(x1, y1) - upper_left
        new_end = Point(x2, y2) - upper_left
        self.set_attrs(x1=new_start.x, y1=new_start.y, x2=new_end.x, y2=new_end.y)
        return self


class Path(ElemSpan):
    """Path class."""

    @cached_property
    def segments_with_bbox(self) -> tuple[list[tuple[str, list[complex]]], BBox]:
        """Cached segments with bbox."""
        return get_segments_with_bbox(self.attr["d"])

    @override
    @property
    def special_border(self) -> bool:
        return True

    @override
    def get_bbox(self) -> BBox:
        return self.segments_with_bbox[1]

    def _get_segments(self) -> list[tuple[str, list[complex]]]:
        return self.segments_with_bbox[0]

    @override
    def adjust(self, upper_left: Point, lower_right: Point) -> Self:
        del lower_right  # only need upper left
        segments = self._get_segments()
        d: list[str] = []
        for segment, points in segments:
            if not points:
                d.append(segment)
                continue

            new_points = [p - upper_left for p in points]
            new_points_str = " ".join([Point(x.real, x.imag).text for x in new_points])
            d.append(f"{segment} {new_points_str}")

        joined_d = " ".join(d)
        self.set_attrs(d=joined_d)
        return self
