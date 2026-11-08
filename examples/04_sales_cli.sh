#!/bin/sh
# Example 4, step 2: the command line on sales.csv, once per report format.
# Writes 04_sales_report/markdown/ and 04_sales_report/latex/. The search is
# seeded, so both runs select the same structure. It stops two levels after
# BIC stops falling: the fifth level finds Lin + WN + Per, the formula the
# data were generated with, and the two after it show nothing better.
#
# Author: Warith Harchaoui <warithmetics@deraison.ai>
set -e
cd "$(dirname "$0")"
[ -f sales.csv ] || python 04_make_sales.py
autostat sales.csv --time month --value units --max-depth 20 --patience 2 --no-check \
    --out 04_sales_report/markdown
autostat sales.csv --time month --value units --max-depth 20 --patience 2 --no-check \
    --format latex --out 04_sales_report/latex
