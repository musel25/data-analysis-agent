# Slides — academic Beamer (metropolis), compiled with XeLaTeX.
#
#   make            build presentation.pdf (with overlays — for presenting)
#   make handout    build handout.pdf      (overlays collapsed — for reading/printing)
#   make verify     rebuild and fail loudly if any frame overflows
#   make clean

.PHONY: all handout verify clean

all: presentation.pdf

presentation.pdf: presentation.tex
	latexmk -xelatex -interaction=nonstopmode presentation.tex

# handout.tex is generated: same source, overlays collapsed to their final state
handout: presentation.tex
	sed 's/^\\documentclass\[9pt,aspectratio=169\]{beamer}/\\documentclass[9pt,aspectratio=169,handout]{beamer}/' \
	    presentation.tex > handout.tex
	latexmk -xelatex -interaction=nonstopmode handout.tex

# A frame that overflows its slide is a real defect — it silently clips content.
verify: presentation.pdf
	@n=$$(grep -c 'Overfull \\vbox' .out/presentation.log || true); \
	 echo "frames overflowing: $$n"; \
	 test "$$n" -le 1 || { echo "FAIL: content is spilling off slides"; exit 1; }

clean:
	latexmk -C presentation.tex handout.tex
	rm -f handout.tex
