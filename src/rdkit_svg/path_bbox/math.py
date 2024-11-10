# %%
"""Mathematical functions for the SVG path utilities."""

# allow mathematical names, which would be invalid otherwise
# ruff: noqa: N803
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import numba
from numba import njit
from numpy import nan

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")

# for easier access
bool_ = numba.types.bool_
f32 = numba.types.float32
c64 = numba.types.complex64
Tuple = numba.types.Tuple

if os.environ.get("COVERAGE_DEBUG", "0") == "1":

    def njit(  # pylint: disable=function-redefined
        *args: Any, **kwargs: Any
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """Dummy decorator if numba is deactivated."""
        del args, kwargs  # as it is just a debug tool, args and kwargs are not used

        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            return func

        return decorator


@njit(f32(f32, f32, f32, f32))
def quadratic_bezier(t: float, P0: float, P1: float, P2: float) -> float:
    """Evaluate the quadratic Bezier curve at t."""
    return (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t**2 * P2


@njit(f32(f32, f32, f32, f32, f32))
def cubic_bezier(t: float, P0: float, P1: float, P2: float, P3: float) -> float:
    """Evaluate the cubic Bezier curve at t."""
    return (
        (1 - t) ** 3 * P0
        + 3 * (1 - t) ** 2 * t * P1
        + 3 * (1 - t) * t**2 * P2
        + t**3 * P3
    )


@njit(f32(f32, f32, f32))
def solve_for_extreme(P0: float, P1: float, P2: float) -> float:
    """Solve for the extreme point of a quadratic Bezier curve.

    1. derivative: 2(1-t)(P1-P0) + 2t(P2-P1) = 0
    => t = (P1-P0) / (P0 - 2P1 + P2)
    """
    denominator = P0 - 2 * P1 + P2
    if denominator == 0:
        return nan
    return (P0 - P1) / denominator


@njit(Tuple([c64, c64])(c64, c64, c64))
def quadratic_bezier_bbox(
    P0: complex, P1: complex, P2: complex
) -> tuple[complex, complex]:
    """Get the bounding box of a quadratic Bezier curve.

    https://www.desmos.com/calculator/fsgcq11iqf
    """
    tx = solve_for_extreme(P0.real, P1.real, P2.real)
    ty = solve_for_extreme(P0.imag, P1.imag, P2.imag)

    # if tx and ty are not in the range [0, 1]
    # P0 and P2 are the bounding box
    points = [P0, P2]

    for t in (tx, ty):
        if not 0 <= t <= 1:
            continue

        points.append(
            complex(
                quadratic_bezier(t, P0.real, P1.real, P2.real),
                quadratic_bezier(t, P0.imag, P1.imag, P2.imag),
            )
        )

    xs = [p.real for p in points]
    ys = [p.imag for p in points]

    return complex(min(xs), min(ys)), complex(max(xs), max(ys))


@njit(Tuple([f32, f32, f32])(f32, f32, f32, f32))
def derivative_coefficients(
    P0: float, P1: float, P2: float, P3: float
) -> tuple[float, float, float]:
    """Get the coefficients of the derivative of the cubic Bezier curve.

    1. derivative: -3(1-t)^2P0 + 3(1-t)^2P1 - 6t(1-t)P1 + 6t(1-t)P2 - 3t^2P2 + 3t^2P3
    to the form: at^2 + bt + c
    gives the coefficients: a, b, c
    """
    return (
        -3 * P0 + 9 * P1 - 9 * P2 + 3 * P3,
        6 * P0 - 12 * P1 + 6 * P2,
        -3 * P0 + 3 * P1,
    )


@njit(Tuple([f32, f32])(f32, f32, f32))
def solve_quadratic_from_coeffs(a: float, b: float, c: float) -> tuple[float, float]:
    """Solve a quadratic equation from the coefficients.

    Returns:
        A tuple with the two solutions of the quadratic equation.
        NaN if there are if a solution is non-real.
    """
    # Solve the quadratic equation ax^2 + bx + c = 0
    # -b ± sqrt(b^2 - 4ac) / 2a
    if a == 0:
        if b == 0:
            return (nan, nan)  # No solution if both `a` and `b` are zero
        # single solution
        return (-c / b, nan)

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        # No solutions
        return (nan, nan)

    sqrt_discriminant = math.sqrt(discriminant)
    t1 = (-b + sqrt_discriminant) / (2 * a)
    t2 = (-b - sqrt_discriminant) / (2 * a)
    # two solutions
    return (t1, t2)


@njit(Tuple([c64, c64])(c64, c64, c64, c64))
def cubic_bezier_bbox(
    P0: complex, P1: complex, P2: complex, P3: complex
) -> tuple[complex, complex]:
    """Get the bounding box of a cubic Bezier curve.

    https://www.desmos.com/calculator/ifyeddi2eh
    """
    # Solve for critical points in x and y directions

    # get the coefficients of the derivative of the cubic Bezier curve
    # in the form at^2 + bt + c
    dx_coeffs = derivative_coefficients(P0.real, P1.real, P2.real, P3.real)
    dy_coeffs = derivative_coefficients(P0.imag, P1.imag, P2.imag, P3.imag)

    txs = solve_quadratic_from_coeffs(*dx_coeffs)
    tys = solve_quadratic_from_coeffs(*dy_coeffs)

    # Evaluate the cubic Bezier curve at t = 0, t = 1, and at the critical points
    x_points = [P0.real, P3.real] + [
        cubic_bezier(t, P0.real, P1.real, P2.real, P3.real) for t in txs if 0 <= t <= 1
    ]
    y_points = [P0.imag, P3.imag] + [
        cubic_bezier(t, P0.imag, P1.imag, P2.imag, P3.imag) for t in tys if 0 <= t <= 1
    ]

    return complex(min(x_points), min(y_points)), complex(max(x_points), max(y_points))


@njit(Tuple([c64, c64])(c64, c64, c64, f32, bool_, bool_))
def arc_bbox(
    start: complex,
    end: complex,
    radii: complex,
    x_axis_rotation: float,
    large_arc: bool,
    sweep: bool,
) -> tuple[complex, complex]:
    """Get the bounding box of an arc."""
    # Convert rotation angle from degrees to radians
    rx, ry = radii.real, radii.imag

    phi = math.radians(x_axis_rotation)

    # Step 1: Compute (x1', y1')
    x1p = (
        math.cos(phi) * (start.real - end.real) / 2
        + math.sin(phi) * (start.imag - end.imag) / 2
    )
    y1p = (
        -math.sin(phi) * (start.real - end.real) / 2
        + math.cos(phi) * (start.imag - end.imag) / 2
    )

    # Step 2: Compute (cx', cy')
    rx_sq, ry_sq = rx**2, ry**2
    x1p_sq, y1p_sq = x1p**2, y1p**2

    radical = max(
        0,
        (rx_sq * ry_sq - rx_sq * y1p_sq - ry_sq * x1p_sq)
        / (rx_sq * y1p_sq + ry_sq * x1p_sq),
    )
    coefficient = math.sqrt(radical)
    if large_arc == sweep:
        coefficient = -coefficient
    cxp = coefficient * rx * y1p / ry
    cyp = -coefficient * ry * x1p / rx

    # Step 3: Compute (cx, cy) from (cx', cy')
    cx = math.cos(phi) * cxp - math.sin(phi) * cyp + (start.real + end.real) / 2
    cy = math.sin(phi) * cxp + math.cos(phi) * cyp + (start.imag + end.imag) / 2

    # Step 4: Compute start and end angles
    start_angle = math.atan2((y1p - cyp) * rx, (x1p - cxp) * ry)
    end_angle = math.atan2((-y1p - cyp) * rx, (-x1p - cxp) * ry)

    # Ensure end_angle > start_angle
    if sweep and end_angle < start_angle:
        end_angle += 2 * math.pi
    elif not sweep and end_angle > start_angle:
        end_angle -= 2 * math.pi

    # Compute extreme points
    extreme_angles = [0, math.pi / 2, math.pi, 3 * math.pi / 2]
    min_x, min_y = start.real, start.imag
    max_x, max_y = start.real, start.imag

    for angle in extreme_angles:
        if start_angle < angle < end_angle or end_angle < angle < start_angle:
            x = (
                cx
                + rx * math.cos(angle) * math.cos(phi)
                - ry * math.sin(angle) * math.sin(phi)
            )
            y = (
                cy
                + rx * math.cos(angle) * math.sin(phi)
                + ry * math.sin(angle) * math.cos(phi)
            )
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # Include end point in bounding box
    min_x = min(min_x, end.real)
    min_y = min(min_y, end.imag)
    max_x = max(max_x, end.real)
    max_y = max(max_y, end.imag)

    return complex(min_x, min_y), complex(max_x, max_y)
