"""Tests the path bbox calculation."""

from __future__ import annotations

import timeit

import numpy as np
import pytest
import svgpathtools

from rdkit_svg.path_bbox import get_segments_with_bbox


def _bbox(test_input: str) -> None:
    """Compare the bbox with svgpathtools' bbox."""
    # calc is b box (xmin, ymin, width, height)
    calc = get_segments_with_bbox(test_input)[1]
    # expected is b box (xmin, xmax, ymin, ymax)

    expected = svgpathtools.parse_path(test_input).bbox()

    converted = (
        expected[0],
        expected[2],
        expected[1] - expected[0],
        expected[3] - expected[2],
    )

    # float32: error range
    np.testing.assert_almost_equal(np.array(calc), np.array(converted), decimal=5)


@pytest.mark.parametrize(
    "test_input",
    [
        "M 219.8 10.4 Q 219.8 9.1 220.4 8.3",
        "M 219.8 10.4 Q 219.8 9.1 220.4 8.3 Q 221.1 7.5 222.4 7.5",
    ],
)
def test_bbox(test_input: str) -> None:
    # M 219.8 10.4 Q 219.8 9.1 220.4 8.3 Q 221.1 7.5 222.4 7.5 Q 223.6 7.5 224.3 8.3 Q 225.0 9.1 225.0 10.4 Q 225.0 11.8 224.3 12.6 Q 223.6 13.4 222.4 13.4 Q 221.1 13.4 220.4 12.6 Q 219.8 11.8 219.8 10.4 M 222.4 12.7 Q 223.2 12.7 223.7 12.2 Q 224.2 11.6 224.2 10.4 Q 224.2 9.3 223.7 8.7 Q 223.2 8.2 222.4 8.2 Q 221.5 8.2 221.0 8.7 Q 220.6 9.3 220.6 10.4 Q 220.6 11.6 221.0 12.2 Q 221.5 12.7 222.4 12.7 # noqa: E501
    _bbox(test_input)


cubics = [
    "M 10 10 C 20 20, 40 20, 50 10",
    "M 70 10 C 70 20, 110 20, 110 10",
    "M 130 10 C 120 20, 180 20, 170 10",
    "M 10 60 C 20 80, 40 80, 50 60",
    "M 70 60 C 70 80, 110 80, 110 60",
    "M 130 60 C 120 80, 180 80, 170 60",
    "M 10 110 C 20 140, 40 140, 50 110",
    "M 70 110 C 70 140, 110 140, 110 110",
    "M 130 110 C 120 140, 180 140, 170 110",
]

quads = ["M 10 80 Q 95 10 180 80", "M 10 80 Q 52.5 10, 95 80 T 180 80"]

relative = [
    "m 10 350 l 40 0 l 20 50",
    "m 70 350 l 40 0 l 20 50",
    "m 130 350 l 40 0 l 20 50",
    "m 10 450 q 30 -40, 60 0 t 120 0",
    "m 70 450 q 30 -40, 60 0 t 120 0",
    "m 130 450 q 30 -40, 60 0 t 120 0",
    "m 10 500 l 50 0 l 20 -50 l 20 50 l 50 0",
    "m 10 600 q 50 -50, 100 0 q 50 50, 100 0",
]


@pytest.mark.parametrize("test_input", relative)
def test_relative(test_input: str) -> None:
    _bbox(test_input)


@pytest.mark.parametrize("test_input", cubics)
def test_cubic(test_input: str) -> None:
    _bbox(test_input)


@pytest.mark.parametrize("test_input", quads)
def test_quad(test_input: str) -> None:
    _bbox(test_input)


def test_moveto() -> None:
    with pytest.raises(ValueError, match="Only move commands found in path data"):
        _bbox("M 10 10")


@pytest.mark.parametrize(
    "test_input",
    [
        "M 10 10 H 50",
        "M 10 10 h 40",
        "M 10 10 V 50",
        "M 10 10 v 40",
        "M 10 10 H 50 V 50 H 10 V 10",
        "M 10 10 h 40 v 40 h -40 v -40",
    ],
)
def test_vertical_horizontal(test_input: str) -> None:
    _bbox(test_input)


arcs = [
    "M 10 315 A 15 15 0 0 1 40 315",
    "M 70 315 A 15 15 0 1 1 100 315",
    "M 130 315 A 15 15 0 0 0 160 315",
    "M 10 365 A 15 15 0 1 0 40 365",
    "M 70 365 a 15 15 0 0 1 30 0",
    "M 130 365 a 15 15 0 1 1 30 0",
]


@pytest.mark.parametrize("test_input", arcs)
def test_arc(test_input: str) -> None:
    _bbox(test_input)


# Add these new test cases after the existing 'quads' list
smooth_curves = [
    "M 10 80 Q 52.5 10 95 80 T 180 80",
    "M 10 180 C 40 100, 65 100, 95 180 S 150 260, 180 180",
    "M 10 280 Q 52.5 210 95 280 T 180 280 T 265 280",
    "M 10 380 C 40 300, 65 300, 95 380 S 150 460 180 380 S 265 300 295 380",
]


@pytest.mark.parametrize("test_input", smooth_curves)
def test_smooth_curves(test_input: str) -> None:
    _bbox(test_input)


all_inputs = relative + cubics + quads + arcs + smooth_curves


@pytest.mark.timeout(0)
def test_bbox_performance() -> None:
    def our_bbox() -> None:
        for path in all_inputs:
            get_segments_with_bbox(path)

    def svgpathtools_bbox() -> None:
        for path in all_inputs:
            svgpathtools.parse_path(path).bbox()

    impl = timeit.timeit(our_bbox, number=1000)
    pathtools_impl = timeit.timeit(svgpathtools_bbox, number=1000)

    assert impl < pathtools_impl
