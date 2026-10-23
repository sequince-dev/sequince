.PHONY: joss arxiv

# Build the JOSS PDF (default).
joss:
	docker run --rm \
		--volume "$(CURDIR):/data" \
		--user "$$(id -u):$$(id -g)" \
		--env JOURNAL=joss \
		openjournals/inara \
		-o pdf paper.md

# Build the arXiv preprint.
arxiv:
	docker run --rm \
		--volume "$(CURDIR):/data" \
		--user "$$(id -u):$$(id -g)" \
		--env JOURNAL=joss \
		openjournals/inara \
		-o preprint paper.md
