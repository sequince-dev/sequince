.PHONY: joss arxiv arxiv.pdf

# Build the JOSS PDF without the abstract (default).
joss:
	docker run --rm \
		--volume "$(CURDIR):/data" \
		--user "$$(id -u):$$(id -g)" \
		--env JOURNAL=joss \
		openjournals/inara \
		-o pdf -- paper.md --metadata=abstract:

# Build the arXiv preprint.
arxiv:
	docker run --rm \
		--volume "$(CURDIR):/data" \
		--user "$$(id -u):$$(id -g)" \
		--env JOURNAL=joss \
		openjournals/inara \
		-o preprint paper.md

# Compile the arXiv preprint to paper.preprint.pdf.
arxiv.pdf: arxiv
	docker run --rm \
		--volume "$(CURDIR):/data" \
		--user "$$(id -u):$$(id -g)" \
		--workdir /data \
		--env XDG_CACHE_HOME=/tmp \
		--entrypoint latexmk \
		openjournals/inara \
		-xelatex -interaction=nonstopmode -halt-on-error paper.preprint.tex
