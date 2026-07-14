# Slides — traditional academic Beamer (Madrid + Palatino), compiled with pdfLaTeX.
#
#   make            build presentation.pdf (with overlays — for presenting)
#   make handout    build handout.pdf      (overlays collapsed — for reading/printing)
#   make verify     rebuild and fail loudly if any frame overflows its slide
#   make clean

.PHONY: all handout verify clean

all: presentation.pdf

presentation.pdf: presentation.tex
	latexmk -pdf -interaction=nonstopmode presentation.tex

# handout.tex is generated: same source, overlays collapsed to their final state
handout: presentation.tex
	sed 's/^\\documentclass\[9pt,aspectratio=169\]{beamer}/\\documentclass[9pt,aspectratio=169,handout]{beamer}/' \
	    presentation.tex > handout.tex
	latexmk -pdf -interaction=nonstopmode handout.tex

# A frame that overflows its slide is a real defect — it silently clips content off the bottom.
verify: presentation.pdf
	@n=$$(grep -c 'Overfull \\vbox' .out/presentation.log || true); \
	 echo "frames overflowing: $$n"; \
	 test "$$n" -le 0 || { echo "FAIL: content is spilling off slides — grep 'Overfull \\vbox' .out/presentation.log"; exit 1; }

clean:
	latexmk -C presentation.tex handout.tex
	rm -f handout.tex
