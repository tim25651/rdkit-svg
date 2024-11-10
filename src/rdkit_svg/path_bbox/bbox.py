"""Parse SVG path data and calculate the bounding box."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypeGuard

from .constants import (
    SPLIT_COUNTS,
    SPLIT_PATTERN,
    SUBCOMMAND_PATTERN,
    VALID_COMMANDS,
    ValidCommand,
)
from .math import arc_bbox, cubic_bezier_bbox, quadratic_bezier_bbox

if TYPE_CHECKING:
    from collections.abc import Sequence

BBox: TypeAlias = tuple[float, float, float, float]
"""Bounding box as a tuple (x, y, width, height)."""


def _check_subcommands(
    data: list[tuple[str, bool, str]],
) -> TypeGuard[list[tuple[ValidCommand, bool, str]]]:
    """Check if the subcommands are valid."""
    subcommands = {x[0] for x in data}

    if subcommands == {"M"}:
        raise ValueError("Only move commands found in path data")

    if invalid := subcommands - VALID_COMMANDS:
        raise ValueError(f"Invalid subcommands found in path data: {invalid}")

    return True


def _split_in_subcommands(d: str) -> list[tuple[ValidCommand, bool, str]]:
    """Split the path data into subcommands and their data.

    Raises:
        ValueError: If no subcommands are found in the path data.
        ValueError: If only move commands are found in the path data.
        NotImplementedError: If any of the subcommands are not implemented.
    """
    subcommands_with_data: list[tuple[str, str]] = SUBCOMMAND_PATTERN.findall(d)

    if not subcommands_with_data:
        raise ValueError("No subcommands found in path data")

    subcommands_with_info = [
        (x.upper(), x.islower(), y) for x, y in subcommands_with_data
    ]

    if not _check_subcommands(subcommands_with_info):
        raise ValueError("Invalid subcommands found in path data")

    return subcommands_with_info


def _split_in_points(
    command: ValidCommand, is_rel: bool, data: str, curr_pos: complex
) -> list[complex]:
    """Split the data of a subcommand into points."""
    values = SPLIT_PATTERN.split(data.strip(), maxsplit=SPLIT_COUNTS[command])
    floats = [float(x) for x in values]

    points = [complex(floats[i], floats[i + 1]) for i in range(0, len(floats), 2)]

    if not is_rel:
        return points

    return [p + curr_pos for p in points]


def _parse_vertical_horizontal(
    command: ValidCommand, is_rel: bool, data: str, curr_pos: complex
) -> tuple[complex, float]:
    """Parse vertical and horizontal commands."""
    value = float(data.strip())

    if command == "V":
        if is_rel:
            return complex(curr_pos.real, curr_pos.imag + value), value
        return complex(curr_pos.real, value), value

    if is_rel:
        return complex(curr_pos.real + value, curr_pos.imag), value

    return complex(value, curr_pos.imag), value


def _parse_arc(
    is_rel: bool, data: str, curr_pos: complex
) -> tuple[tuple[complex, complex], complex, list[complex | float | bool]]:
    """Parse the arc data."""
    values: list[str] = SPLIT_PATTERN.split(data.strip(), maxsplit=7)
    rx, ry, angle_str, large_arc_str, sweep_str, x, y = values

    radii = complex(float(rx), float(ry))
    angle = float(angle_str)
    if {large_arc_str, sweep_str} - {"0", "1"}:
        raise ValueError("Invalid large arc or sweep value")

    large_arc = large_arc_str == "1"
    sweep = sweep_str == "1"

    end = complex(float(x), float(y))
    if is_rel:
        end += curr_pos

    return (
        arc_bbox(curr_pos, end, radii, angle, large_arc, sweep),
        end,
        [radii, angle, large_arc, sweep, end],
    )


def _parse_quadratic_cubic(
    command: ValidCommand,
    last_command: ValidCommand,
    is_rel: bool,
    data: str,
    curr_pos: complex,
    prev_control: complex | None,
) -> tuple[tuple[complex, complex], list[complex], complex, complex]:
    """Parse quadratic and cubic commands."""
    points = _split_in_points(command, is_rel, data, curr_pos)
    save_points = points

    if command == "T" or command == "S":  # noqa: PLR1714
        prev_commands = {"Q", "T"} if command == "T" else {"C", "S"}

        if prev_control and last_command in prev_commands:
            # Reflect previous control point
            control = 2 * curr_pos - prev_control
        else:
            control = curr_pos

        points = [control, *points]

    if command == "Q" or command == "T":  # noqa: PLR1714
        bbox = quadratic_bezier_bbox(curr_pos, *points)
        prev_control = points[0]
    else:
        bbox = cubic_bezier_bbox(curr_pos, *points)
        prev_control = points[1]

    curr_pos = points[-1]

    return bbox, save_points, curr_pos, prev_control


def get_bbox(points: Sequence[complex]) -> BBox:
    """Calculates the bounding box from multiple points.

    Args:
        points: A list of points as complex numbers.

    Returns:
        The bounding box as a tuple (x, y, width, height).
    """
    if not points:
        raise ValueError("No bounding box points found")

    x = min(x.real for x in points)
    y = min(x.imag for x in points)
    width = max(x.real for x in points) - x
    height = max(x.imag for x in points) - y

    return x, y, width, height


def get_segments_with_bbox(
    d: str,
) -> tuple[list[tuple[str, list[complex | float | bool]]], BBox]:
    """Parses the segments of the path data and calculates the bounding box.

    Args:
        d: The path string

    Returns:
        A tuple with the segments of the path data and the bounding box
        (x, y, width, height).

    Example:
        >>> get_segments_with_bbox("M 10 10 L 20 20 L 10 30 Z")
        (
        [('M', [(10+10j)]), ('L', [(20+20j)]), ('L', [(10+30j)]), ('Z', [])],
        (10, 10, 10, 20)
        )
    """
    final: list[tuple[str, list[complex | float | bool]]] = []

    all_bbox_points: list[complex] = []

    curr_pos = complex(0, 0)
    prev_control: complex | None = None

    subcommands = _split_in_subcommands(d)

    final_ix = len(subcommands) - 1

    for ix, (command, is_rel, data) in enumerate(subcommands):
        if command == "M":
            curr_pos = _split_in_points(command, is_rel, data, curr_pos)[0]
            final.append((command, [curr_pos]))

        elif command == "Z":
            if ix != final_ix:
                raise ValueError("Z command should be at the end of the path data")
            final.append((command, []))

        elif command == "V" or command == "H":  # noqa: PLR1714
            point, value = _parse_vertical_horizontal(command, is_rel, data, curr_pos)
            all_bbox_points.extend((curr_pos, point))
            curr_pos = point
            final.append((command, [value]))

        elif command == "L":
            point = _split_in_points(command, is_rel, data, curr_pos)[0]
            all_bbox_points.extend((curr_pos, point))
            curr_pos = point
            final.append((command, [curr_pos]))

        elif command == "A":
            bbox, curr_pos, values = _parse_arc(is_rel, data, curr_pos)

            all_bbox_points.extend(bbox)
            final.append((command, values))

        else:
            last_command = subcommands[ix - 1][0]
            bbox, save_points, curr_pos, prev_control = _parse_quadratic_cubic(
                command, last_command, is_rel, data, curr_pos, prev_control
            )

            all_bbox_points.extend(bbox)
            final.append((command, save_points))

    return final, get_bbox(all_bbox_points)


# TODO: rewrite as classes and make bounding box calculation independent of
# parsing the path data
