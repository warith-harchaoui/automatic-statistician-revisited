# Contributing

[CODING.md](CODING.md) is the engineering standard of this repository: read
it before editing. Prose follows two more rules, in English and in French
alike. `scripts/lint_prose.py` enforces them in every text file and every
code comment:

- no em dash and no en dash;
- no comma before "and" or "et".

## Set up

```bash
pip install -r requirements-dev.txt
make check
```

`make check` runs the formatter check, the linter, the prose check and the
fast tests, the same steps as continuous integration (CI). `make test-all`
adds the slow tests. They run real searches and take minutes.

## The rule that matters most

Every number in the documentation comes from a run of this code, with its
date. The likelihood test against the 130 published models is what lets this
port claim fidelity. A change that breaks it is wrong until a measurement
shows otherwise.
