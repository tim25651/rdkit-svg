"""Functions for manipulating SVG trees."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, TypeGuard
from xml.etree import ElementTree as ET

from defusedxml.ElementTree import fromstring

import rdkit_svg.wrappers as svg_classes
from rdkit_svg.wrappers import Box, ElemSpan, Point, filtered_tag

if TYPE_CHECKING:
    from collections.abc import Iterable

Tree: TypeAlias = dict[ET.Element, list["Tree | ET.Element"]]
Seen = set[ET.Element]


def save_parse(data: str) -> ET.Element:
    """Save and parse an SVG string."""
    return fromstring(data)  # type: ignore[no-any-return]


def assure_elem(elem: Any) -> TypeGuard[ET.Element]:
    """Assure that the element is an `ET.Element`."""
    return isinstance(elem, ET.Element)


def filtered_tag_with_provider(tag: str) -> tuple[str, str]:
    """Get the tag and provider from an XML tag."""
    provider_match = re.search(r"({.*})", tag)
    if provider_match is None:
        raise ValueError(f"Could not find provider in {tag}")
    provider_str = provider_match.group(1)

    return filtered_tag(tag), provider_str


def to_string(elem: ET.Element) -> str:
    """Convert an element to a string."""
    # make as template to make it re
    string = ET.tostring(elem).decode("utf-8")
    string = string.replace("ns0:", "").replace(":ns0", "")
    string = string.replace("svg:", "").replace(":svg", "")
    return string.strip()


def read_tree(data: str | Path) -> ET.Element:
    """Read an SVG tree."""
    if isinstance(data, Path):
        data = data.read_text("utf-8")

    return save_parse(data)


def get_class_from_tag(tag: str) -> type[ElemSpan] | None:
    """Get the class from the tag."""
    wrapped_classes: dict[str, type[ElemSpan]] = {
        "rect": svg_classes.Rect,
        "polygon": svg_classes.Polygon,
        "ellipse": svg_classes.Ellipse,
        "circle": svg_classes.Circle,
        "line": svg_classes.Line,
        "path": svg_classes.Path,
        "text": svg_classes.Text,
        "svg": svg_classes.Svg,
    }
    none_classes: set[str] = {
        "g",
        "metadata",
        "defs",
        "title",
        "desc",
        "clipPath",
        "marker",
        "style",
    }

    if tag in none_classes:
        return None

    if elem := wrapped_classes.get(tag):
        return elem

    raise NotImplementedError(f"Tag {tag} not implemented")


def get_class(elem: ET.Element) -> type[ElemSpan] | None:
    """Get the class from the element."""
    tag, provider = filtered_tag_with_provider(elem.tag)
    if provider != "{http://www.w3.org/2000/svg}":
        return None

    return get_class_from_tag(tag)


def validate_bounds(
    bounds: tuple[Point | None, Point | None],
) -> tuple[Point, Point] | None:
    """Validate the bounds."""
    a, b = bounds
    if a is None and b is None:
        # has no bounds
        return None

    if a is None or b is None:
        raise ValueError("Invalid bounds")

    return a, b


def find_bounds_for_obj(
    obj: ElemSpan,
) -> tuple[None | tuple[ElemSpan, Box], Iterable[Point]]:
    """Find the bounds for an object."""
    maybe_bounds = obj.get_bounds()
    bounds = validate_bounds(maybe_bounds)

    if bounds is None:
        return None, ()

    if obj.has_border and obj.special_border:
        # Extend the bounds by the stroke width to get the maximum bounds
        pad = obj.stroke_padding
        a, b = bounds
        min_box = Box(a + pad, b - pad)
        max_box = Box(a - pad, b + pad)

        return (obj, max_box), min_box

    return None, bounds


def find_bounds(tree: ET.Element) -> tuple[Point, Point]:
    """Find the bounds for an SVG tree."""
    objs: list[ElemSpan] = []

    i_tree = iter_tree(tree)
    if isinstance(i_tree, ET.Element):
        raise TypeError("Tree has no elements")

    for elem, prev in extract_tree(i_tree):
        if {"clipPath", "marker"} & set(prev):
            continue

        cls = get_class(elem)

        if cls is None:
            continue

        obj = cls(elem, tree)

        if obj.hidden:
            # Don't calculate bounds for hidden objects
            continue

        objs.append(obj)

    unsafe_objs_: list[tuple[ElemSpan, Box] | None] = []
    safe_points: list[Point] = []

    for obj in objs:
        unsafe_, safe_ = find_bounds_for_obj(obj)
        unsafe_objs_.append(unsafe_)
        safe_points.extend(safe_)

    unsafe_objs = [x for x in unsafe_objs_ if x is not None]

    safe_points.extend(point for _, max_box in unsafe_objs for point in max_box)

    return Point.min(*safe_points), Point.max(*safe_points)


class ApplyOnObj(Protocol):
    """Apply a function on an `ElemSpan` object."""

    def __call__(self, obj: ElemSpan) -> None:
        """Apply the function."""


def iter_tree(tree: ET.Element, seen: Seen | None = None) -> Tree | ET.Element:
    """Iterate over an SVG tree."""
    if seen is None:
        seen = set()

    seen.add(tree)

    subelements = list(tree.iter())[1:]  # skip self

    if len(subelements) == 0:
        return tree

    return {tree: [iter_tree(y, seen) for y in subelements if y not in seen]}


def extract_tree(
    tree: Tree, prev: tuple[str, ...] | None = None
) -> list[tuple[ET.Element, tuple[str, ...]]]:
    """Extract the tree."""
    if prev is None:
        prev = ()

    data: list[tuple[ET.Element, tuple[str, ...]]] = []

    for key, value in tree.items():
        data.append((key, prev))
        curr = (*prev, filtered_tag(key.tag))

        for elem in value:
            if isinstance(elem, ET.Element):
                data.append((elem, curr))
            else:
                data.extend(extract_tree(elem, curr))

    return data


def build_tree(tree: ET.Element, fn: ApplyOnObj) -> ET.Element:
    """Build the tree."""
    assure_elem(tree)

    i_tree = iter_tree(tree)
    if isinstance(i_tree, ET.Element):
        raise TypeError("Tree has no elements")

    for elem, prev in extract_tree(i_tree):
        if {"clipPath", "marker"} & set(prev):
            continue

        cls = get_class(elem)
        if cls is None:
            continue

        obj = cls(elem, tree)
        fn(obj)

    return tree


def adjust_tree(tree: ET.Element, upper_left: Point, lower_right: Point) -> ET.Element:
    """Adjust the tree to the new bounds."""

    def fn(obj: ElemSpan) -> None:
        obj.adjust(upper_left, lower_right)

    return build_tree(tree, fn)


def remove_whitespace(tree: ET.Element, padding: float = 0) -> ET.Element:
    """Remove the whitespace from an SVG tree."""
    padding_ = Point(padding, padding)

    upper_left, lower_right = find_bounds(tree)

    return adjust_tree(tree, upper_left - padding_, lower_right + padding_)


def hide_objects(tree: ET.Element, tags: Iterable[str]) -> ET.Element:
    """Call the hide method on objects with the given tags."""
    tags_ = set(tags)

    def fn(obj: ElemSpan) -> None:
        if filtered_tag(obj.tag) in tags_:
            obj.hide()

    return build_tree(tree, fn)
