"""Content generators for SynthDoc — text, formulas, tables, figures, handwriting, signatures."""

from synthdoc.content.figure import FigureGenerator
from synthdoc.content.formula import FormulaGenerator
from synthdoc.content.handwriting import HandwritingGenerator
from synthdoc.content.signature import SignatureGenerator
from synthdoc.content.table import TableGenerator
from synthdoc.content.text import TextGenerator

__all__ = [
    "TextGenerator",
    "FormulaGenerator",
    "TableGenerator",
    "FigureGenerator",
    "HandwritingGenerator",
    "SignatureGenerator",
]
