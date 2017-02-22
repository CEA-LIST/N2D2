if not exist build mkdir build
pdflatex -output-directory build manual.tex
copy biblio.bib build
cd build
bibtex manual
cd ..
pdflatex -output-directory build manual.tex
copy build\manual.pdf . /Y
