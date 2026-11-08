#!/bin/sh
# Build the repository report in both formats from one source.
#
# make_figures.py draws the figures (SVG for Markdown, PDF for LaTeX).
#
# markdown/report.md is the source. latex/report.tex is generated from it by
# Pandoc (citations as BibLaTeX), so the two cannot drift apart; then each is
# compiled: the Markdown by md2star (md2pdf), the LaTeX by latexmk with
# LuaLaTeX and biber. Both read references.bib.
#
# Needs: pandoc, md2star (with LibreOffice), a TeX distribution with latexmk,
# LuaLaTeX and biber.
#
# Author: Warith Harchaoui <warithmetics@deraison.ai>
set -e
cd "$(dirname "$0")"
python make_figures.py
pandoc markdown/report.md --standalone --biblatex \
    -V documentclass=article -V geometry:margin=2.5cm -V colorlinks=true \
    --metadata bibliography=../references.bib \
    -o latex/report.tex
# LaTeX includes the PDF twin of each SVG figure.
sed -i.bak -e 's#\(figures/[A-Za-z0-9_]*\)\.svg#\1.pdf#g' \
    -e 's#\\includesvg#\\includegraphics#g' latex/report.tex && rm -f latex/report.tex.bak
(cd markdown && md2pdf report.md --bib ../references.bib -o report.pdf)
(cd latex && latexmk -lualatex -interaction=nonstopmode -halt-on-error report.tex \
    && latexmk -c report.tex && rm -f report.bbl)
