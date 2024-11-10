"""Parse SVG path data and calculate the bounding box."""

from __future__ import annotations

from .bbox import BBox, get_bbox, get_segments_with_bbox

__all__ = ["BBox", "get_bbox", "get_segments_with_bbox"]
