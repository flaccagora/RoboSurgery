#!/bin/bash
# should compile in build directory and save pdf in current directory
# create build directory if not exists

echo "Compiling tesi.tex"
pdflatex tesi.tex
echo "Compiling glossary"
makeglossaries tesi
echo "Bbliography"
biber tesi
echo "Compiling tesi.tex"
pdflatex tesi.tex
echo "OK"


# latexmk tesi.tex