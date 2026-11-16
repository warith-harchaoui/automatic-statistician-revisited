# Do a language model's proposals help the search?

`autostat.propose` lets a language model suggest structures at every level;
the search fits them beside the grammar's neighbours and BIC alone decides.
This measures what that changes, on the two synthetic families of
[context_search](../context_search/README.md): a quarterly cycle; a
yearly cycle with a level jump in mid-2014. Each series was searched four
ways, with the 2014 grammar and light settings (three levels, four random
starts, 100 optimiser iterations, two worker processes):

- `grammar`: the grammar alone;
- `llm_none`: plus five proposals per level from gemma3:12b on the local
  Ollama server, told nothing about the business;
- `llm_true`: plus proposals told the true context (a cycle of 0.25 years;
  a yearly cycle and a reform in mid-2014);
- `llm_wrong`: plus proposals told a misleading one (a cycle of five
  months; a reform in 2012).

The context reaches the language model only, never the search's starts, so
the arms differ by the proposals alone. Three seeds per family, 24 runs.
[run.py](run.py) reproduces it; [results.json](results.json) holds every
row with every proposal and every line the reader rejected.

## BIC of the model kept (lower is better)

| family | seed | grammar | llm_none | llm_true | llm_wrong |
|---|---|---|---|---|---|
| quarterly | 0 | **389.1** | **389.1** | 393.6 | 399.3 |
| quarterly | 1 | 389.9 | **376.4** | 382.0 | 382.0 |
| quarterly | 2 | 432.8 | 429.7 | 473.2 | **409.3** |
| jump | 0 | 494.0 | 434.4 | **421.5** | 470.8 |
| jump | 1 | 474.8 | **398.2** | 400.5 | 452.8 |
| jump | 2 | 489.3 | 489.3 | 443.7 | **440.8** |

## What the kept model says

| | grammar | llm_none | llm_true | llm_wrong |
|---|---|---|---|---|
| quarterly cycle found at 0.25 years | 1 of 3 | 2 of 3 | 2 of 3 | 3 of 3 |
| yearly cycle found (jump family) | 1 of 3 | 3 of 3 | 3 of 3 | 3 of 3 |
| jump found within 0.1 year of its date | 0 of 3 | 2 of 3 | 3 of 3 | 3 of 3 |

- **The jump is where proposals matter.** The grammar alone never finds it
  in three levels: it must first build a changepoint from a smooth term,
  then place it. The language model writes `CP(SE, SE)` and its kin from
  the first level (three to six switches among fifteen proposals when told
  of a reform, none to four when told nothing), so the search starts there.
  BIC falls by 22 to 77 on every jump series but one (`llm_none`, seed 2,
  where no switch was proposed and the grammar's model was kept).
- **On the quarterly cycle, the effect is mixed.** The proposals find the
  cycle more often, yet BIC is worse than the grammar's in 3 runs of 9,
  by 40 at most (`llm_true`, seed 2, where `Per + RQ + SE + WN` was kept
  with a wrong period). Five proposals per level take time and draw the
  greedy search into branches it would not have visited; with four random
  starts, the branch can be worse.
- **The wrong context did little harm, for a reason worth knowing.** The
  language model never wrote a period as a number: told of a five-month
  cycle, it proposed `Per` as when told nothing; the period was set by
  the search's starts. Told of a reform in 2012, it tried to date the
  changepoint (`CP(Per, SE, 2012.00)`), which the reader rejects; the jump
  was found where the data put it, 0.015 to 0.06 years from mid-2014. The
  wrong context still kept a higher BIC than the true one on two jump
  series of three. The proposals carry the shape of the context (a cycle,
  a switch), not its numbers.
- **The model kept was itself a proposal in 6 runs of 18.** Elsewhere the
  winner grew from a proposal through the grammar's rewrites, or came from
  the grammar alone; the record does not trace descent.
- Reading the answers is reliable: 4 lines rejected beside 265 structures read,
  no failure of the model. Of the 4, three dated a switch and one wrote
  `Per(p)` literally.

## Cost

The searches with proposals took 48 to 84 minutes per arm against 22 for
the grammar alone, on a machine shared with other measurements (load
around 100). A level waits for the model's answer; the proposals add
five structures to fit.

## A second run: what it showed about the first

The study was run again once the prompt offered dated switches
([results-dated.json](results-dated.json)). The language model wrote its
dates named, `CP(SE, SE, t=2012.00)`, which the reader then refused
(5 lines); named dates are read since. The run measured something else
instead: the grammar arm, unchanged, did not repeat itself. Quarterly
seed 2 went from 432.8 to 409.6, seed 1 from 389.9 to 388.5.

The cause was numerical, not statistical. On macOS, Apple's Accelerate
solves a triangular system with one right-hand side differently from one
call to the next at some sizes (this series has 96 points), in the last
bits. L-BFGS amplified that near ill-conditioned points: one refit of
`C + WN` from a fixed start ended anywhere between 313 and 348. Every
solve now pads a single column to two, which takes a routine that repeats
itself exactly; the same fit then gives the same bits; a test checks
it.

So both runs above were draws among several. What held in both:

| | grammar | llm_none | llm_true | llm_wrong |
|---|---|---|---|---|
| jump found, first run | 0 of 3 | 2 of 3 | 3 of 3 | 3 of 3 |
| jump found, second run | 0 of 3 | 3 of 3 | 2 of 3 | 3 of 3 |

The jump result stands: the grammar alone missed it six times of six, the
proposals found it 16 times of 18. The quarterly differences between arms
(up to 40 BIC in one run) are of the size of the run-to-run variation (23
BIC on the grammar arm alone) and say nothing either way.

## Third run: the dates reach the search

Run again once named dates were read and a fit repeated itself
([results-named.json](results-named.json)). The language model now dated
most of its switches; the dates it wrote were those it was told:
2014.5 when told the truth, 2012 when told the wrong date; guesses
from 2010 to 2017 when told nothing. No dated line was refused.

| jump family, seed | grammar | llm_none | llm_true | llm_wrong |
|---|---|---|---|---|
| 0 | 494.0 | 485.9 | **418.7** | 486.0 |
| 1 | 474.8 | 391.8 | **387.8** | 394.2 |
| 2 | 489.3 | 500.2 | **428.5** | 463.8 |
| mean | 486.0 | 459.3 | **411.7** | 448.0 |

- **The true date helps, a wrong date now costs.** Told the truth, the
  search kept the lowest BIC on all three series, with the jump 0.004 to
  0.083 years from its date. Told 2012, it missed the jump on seed 0 and
  kept a window with an extra changepoint on seed 2: 67 and 35 above the
  true context. In the second run, when dates did not reach the starts,
  the wrong context scored as well as the true one or better on every
  series.
- **BIC still refused the wrong date.** Where a jump was kept, it sat at
  the true date, never at 2012: a start at the wrong date either climbed
  to the right one or lost to a model with no jump.
- **Guessed dates mislead too.** Told nothing, the model invented dates;
  on seeds 0 and 2 the search kept a window instead of the step; on
  seed 2 its BIC was worse than the grammar alone (500.2 against 489.3).
- The grammar arm now repeats itself; its values differ from the first
  two runs where the fix changed the arithmetic (quarterly seeds 1 and 2).
  The language-model arms still vary between runs, the model sampling at
  temperature 0.7.

So a context's numbers are worth giving when they are right; the
prompt should not invite dates nobody gave.

## What follows

- Proposals that date a switch, `CP(a, b, 2014.5)` or
  `CP(a, b, t=2014.5)`, start its location there, as `Per(p)` starts a
  period. The third run measured what a wrong date then costs.
- Three seeds per family is a first look. The jump result is large and
  consistent; the quarterly result is not distinguishable from noise.
