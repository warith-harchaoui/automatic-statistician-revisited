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

## What follows

- Proposals that date a switch, `CP(a, b, 2014.5)`, could start its
  location as `Per(p)` starts a period: the language model already writes
  them so.
- Three seeds per family is a first look. The jump result is large and
  consistent; the quarterly result is not distinguishable from noise.
