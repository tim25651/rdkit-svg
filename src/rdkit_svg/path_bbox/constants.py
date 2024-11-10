"""Constants for the SVG path utilities."""

from __future__ import annotations

import re
from typing import Literal, TypeAlias

COMMANDS = r"MLQCZVHTSAmlqczvhtsa"
"""A string containing all the valid SVG path commands."""

VALID_COMMANDS = set("MLQCZVHTSA")
"""A set containing all the valid upper case SVG path commands."""

ValidCommand: TypeAlias = Literal["M", "L", "Q", "C", "Z", "V", "H", "T", "S", "A"]
"""A type alias for the valid SVG path commands."""

SUBCOMMAND_PATTERN = re.compile(r"([" + COMMANDS + r"])([^" + COMMANDS + r"]+)?")
"""A regex pattern to match SVG path subcommands."""

SPLIT_PATTERN = re.compile(r"[,\s]+")
"""A regex pattern to split the data of a subcommand."""

SPLIT_COUNTS: dict[ValidCommand, int] = {
    "M": 1,
    "L": 1,
    "Q": 3,
    "C": 5,
    "Z": 0,
    "V": 0,
    "H": 0,
    "T": 1,
    "S": 3,
    "A": 6,
}
"""The number of values expected for each SVG path command."""
