"""Draw molecules and convert to SVG."""

from __future__ import annotations

from functools import lru_cache
from io import StringIO
from math import ceil
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.Draw import rdMolDraw2D

from rdkit_svg.utils import (
    assure_elem,
    hide_objects,
    read_tree,
    remove_whitespace,
    to_string,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def get_rdkit_svg(d2d: rdMolDraw2D.MolDraw2DSVG) -> str:
    """Get the SVG string from a finished RDKit drawing."""
    svg = d2d.GetDrawingText()
    # fix for "...</tag>\whitespace</svg>\whitespace"
    return svg.strip().removesuffix("</svg>").strip() + "</svg>"


def draw_with_rdkit(mol: Mol | str, sub_img_size: tuple[int, int] = (250, 200)) -> str:
    """Draw a molecule using RDKit and return the SVG string."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    d2d = rdMolDraw2D.MolDraw2DSVG(*sub_img_size)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()

    return get_rdkit_svg(d2d)


def add_legend(svg: str, legend: str | None = None, line_breaks: bool = True) -> str:
    """Add a legend to the SVG."""
    if legend:
        new_line = "\n<br>\n" if line_breaks else "<br>"
        svg += new_line + legend

    return svg


@lru_cache(maxsize=2048 * 8)
def _cached_rdkit_svg(
    canonical_smi: str, sub_img_size: tuple[int, int] = (250, 200)
) -> str:
    """Cache the SVG for a molecule."""
    mol = Chem.MolFromSmiles(canonical_smi)
    svg = draw_with_rdkit(mol, sub_img_size)
    tree = fix_svg(svg)
    tree, second_tree = make_svg_as_template(canonical_smi, tree)
    tree_str = to_string(tree)
    # remove <svg xmlns="http://www.w3.org/2000/svg">
    tree_str = tree_str[40:]
    second_tree_str = to_string(second_tree)

    return f'<svg hidden="true">{tree_str}<!--template-->{second_tree_str}'


def extract_svg(svg: str) -> tuple[str, str, str]:
    """Get canonical smiles, template and use strings."""
    template, use = svg.split("<!--template-->")

    start_ix = len('<svg hidden="true"><symbol id="')
    end_ix = template.find('"', start_ix)
    key = template[start_ix:end_ix].replace("%%", "#")

    return key, template, use


def _mol_to_svg(mol: str | Mol, sub_img_size: tuple[int, int] = (250, 200)) -> str:
    """Convert a molecule to SVG."""
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    canonical_smi = Chem.MolToSmiles(mol)
    return _cached_rdkit_svg(canonical_smi, sub_img_size)


def mol_to_svg(
    mol: str | Mol,
    legend: str | None = None,
    sub_img_size: tuple[int, int] = (250, 200),
    line_breaks: bool = True,
) -> str:
    """Convert a molecule to SVG."""
    svg = _mol_to_svg(mol, sub_img_size)
    return add_legend(svg, legend, line_breaks)


def mols_to_grid(
    mols: Sequence[Mol | str],
    legends: Sequence[str] | None = None,
    n_per_row: int = 4,
    sub_img_size: tuple[int, int] = (250, 200),
) -> str:
    """Convert a list of molecules to a HTML grid."""
    n_per_row = min(n_per_row, len(mols))
    n_rows = ceil(len(mols) / n_per_row)

    matrix = np.empty((n_rows, n_per_row), dtype=object)
    matrix[:] = ""

    templates: dict[str, str] = {}

    for i, mol in enumerate(mols):
        col = i % n_per_row
        row = i // n_per_row

        legend = legends[i] if legends else None
        svg = _mol_to_svg(mol, sub_img_size)
        key, template, use = extract_svg(svg)
        templates[key] = template
        use = add_legend(use, legend, line_breaks=False)
        matrix[row, col] = use

    # numpy matrix to html table

    buffer = StringIO()

    pd.DataFrame(matrix).to_html(
        buffer, border=1, escape=False, header=None, index=None
    )

    # max-width: 300px;

    svg = buffer.getvalue()

    return (
        "<style>"
        "table {border-collapse:collapse;}"
        "td {max-height:300px; max-width:500px;}"
        "</style>"
        # all in one svg to avoid multiple svg tags
        + "".join(templates.values()).replace('</svg><svg hidden="true">', "")
        + svg
    )


def make_svg_as_template(
    canonical_smi: str, tree: ET.Element
) -> tuple[ET.Element, ET.Element]:
    """Make a copy of the SVG as a template."""
    # <svg attr.... -> <svg hidden><symbol id="CANONICAL">{content}</symbol></svg>
    orig_attrib = tree.attrib.copy()
    # pack into symbol
    symbol = ET.Element("symbol", {"id": canonical_smi.replace("#", "%%")})
    # pack the sub elements from svg into symbol and put symbol into svg
    for child in list(tree):
        symbol.append(child)
    tree.clear()
    tree.append(symbol)
    use = ET.Element("use", {"xlink:href": f"#{canonical_smi.replace('#', '%%')}"})
    second_tree = ET.Element("svg", orig_attrib)
    second_tree.append(use)

    return tree, second_tree


def fix_svg(svg: str | Path) -> ET.Element:
    """Removes whitespace and hidden objects."""
    tree = read_tree(svg)

    if not assure_elem(tree):
        raise ValueError("No root element found")

    hide_objects(tree, ["rect"])
    remove_whitespace(tree)

    return tree


def create_grid(
    data: Sequence[object | Mol | str],
    cols: int = 2,
    smi_attr: str = "smi",
    other_attrs: Sequence[str] | None = None,
) -> str:
    """Add labels to the grid visualization.

    Special case:
        If provided with a object with a `attr` attribute, it will be
        convert it to a `Chem.Mol` object. All other attributes of
        the object will be added to the description.

    Args:
        data: The data to visualize. Either RDKit molecules, SMILES strings,
            or objects with `attr` attribute.
        cols: The number of columns.
        save: The path to save the visualization.
        smi_attr: The attribute to use for the SMILES string (or RDKit mol).
            Defaults to `smi`.
        other_attrs: Other attributes to include in the description. Defaults to None.
    """

    def validate_obj(obj: object) -> tuple[Mol, dict[str, str]]:
        if not hasattr(obj, smi_attr):
            raise ValueError(f"Object {obj} has no attribute {smi_attr}.")
        smi = getattr(obj, smi_attr)
        if not isinstance(smi, str | Mol):
            raise TypeError(f"Attribute {smi_attr} is not a string or RDKit Mol.")

        if other_attrs:
            if not all(hasattr(obj, attr) for attr in other_attrs):
                raise ValueError(f"Object {obj} is missing attributes {other_attrs}.")

            props: dict[str, str] = {
                attr: str(getattr(obj, attr)) for attr in other_attrs
            }
        else:
            props = {}

        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else smi

        return mol, props

    mols: list[Mol] = []
    legends: list[str] = []
    for elem in data:
        if isinstance(elem, Mol):
            mols.append(elem)
            legends.append(Chem.MolToSmiles(elem))
        elif isinstance(elem, str):
            mols.append(Chem.MolFromSmiles(elem))
            legends.append(elem)
        else:
            mol, props = validate_obj(elem)
            mols.append(mol)
            desc = "<br>".join(f"{k.capitalize()}: {v}" for k, v in props.items())
            legends.append(desc)

    return mols_to_grid(mols, n_per_row=cols, legends=legends)


# %%
