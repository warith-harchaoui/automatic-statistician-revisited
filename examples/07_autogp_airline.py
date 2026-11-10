"""Example 7: an AutoGP posterior on the airline series, described like a 2014 model.

Writes 07_autogp_report/markdown/ and 07_autogp_report/latex/: in each, a
summary page with the posterior's structures and weights, plus one
folder per described structure.

Author
------
Warith Harchaoui <warithmetics@deraison.ai>
"""

from pathlib import Path

from autostat import benchmarks
from autostat.smc import AutoGP, linear_schedule
from autostat.smc.report import describe_posterior

OUT = Path(__file__).parent / "07_autogp_report"

x, y = benchmarks.load("01-airline")
model = AutoGP(x.ravel(), y.ravel(), n_particles=8, seed=0)
model.fit_smc(linear_schedule(len(x), 0.10), n_mcmc=75, n_hmc=10)
for weight, formula in model.structures()[:3]:
    print(f"{weight:.3f}  {formula}")
for report_format in ("markdown", "latex"):
    posterior = describe_posterior(
        model,
        name="airline passengers",
        out_dir=OUT / report_format,
        report_format=report_format,
    )
for summary, _, _ in posterior.reports[0].texts:
    print(summary)
