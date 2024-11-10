"""Test the :func:`svg_helper.fix_svg` function."""

from __future__ import annotations

from typing import TYPE_CHECKING

from IPython.core.display import SVG
from rdkit import Chem
from rdkit.Chem import Draw

from rdkit_svg import mol_to_svg, mols_to_grid
from rdkit_svg.draw import fix_svg
from rdkit_svg.utils import to_string

if TYPE_CHECKING:
    from pathlib import Path


def test_mol_to_svg(tmp_path: Path) -> None:
    mol = Chem.MolFromSmiles("C1CCCCC1")
    svg = mol_to_svg(mol)

    file = tmp_path / "mol.svg"
    file_with_legend = tmp_path / "mol_with_legend.svg"

    svg_with_legend = mol_to_svg(mol, "C1CCCCC1")
    file.write_text(svg)
    file_with_legend.write_text(svg_with_legend)


def test_crop(tmp_path: Path) -> None:
    mol = Chem.MolFromSmiles("C1CCCCC1")
    img = Draw.MolsToGridImage([mol], useSVG=True)  # type: ignore[no-untyped-call]
    if isinstance(img, SVG):
        img = img.data

    file = tmp_path / "uncropped.svg"
    file.write_text(img)

    tree = fix_svg(file)
    fixed_svg = to_string(tree)

    save_path = tmp_path / "cropped.svg"
    save_path.write_text(fixed_svg)


def test_mols_to_grid(tmp_path: Path) -> None:
    mols = [Chem.MolFromSmiles("C1CCCCC1" + n * "C") for n in range(10)]
    legends = [f"Legend {n}" for n in range(10)]
    grid = mols_to_grid(mols, legends)

    file = tmp_path / "grid.html"
    file.write_text(grid)
