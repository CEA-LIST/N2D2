compile :
	mkdir -p build
	pdflatex -output-directory build manual.tex
	cp biblio.bib build
	cd build ; bibtex manual
	pdflatex -output-directory build manual.tex

clean :
	rm -rf build
