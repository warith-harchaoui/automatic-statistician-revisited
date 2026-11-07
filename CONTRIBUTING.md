# Contributing

Read [CODING.md](CODING.md) before editing: it is the engineering standard of
this repository. Prose, in English and in French, follows two further rules
that `scripts/lint_prose.py` enforces in every text file and every code
comment: no em dash or en dash; no comma before "and" or "et".

## Set up

```bash
pip install -r requirements-dev.txt
make check
```

`make check` runs the formatter check, the linter, the prose check and the
fast tests: the same steps as continuous integration (CI). `make test-all`
adds the slow tests, which run real searches and take minutes.

## The rule that matters most

Every number in the documentation comes from a run of this code, with its
date. The likelihood test against the 130 published models is the reason
this port can claim fidelity; a change that breaks it is wrong until proven
otherwise.
