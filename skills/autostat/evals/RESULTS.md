# Trigger measurement

Measured on 10 November 2026 with Claude Code 2.1.311, by
[measure_triggers.py](measure_triggers.py): each query of
[triggers.yaml](triggers.yaml) in a fresh headless conversation, the skill
installed as a project skill, alongside the user's other skills.

- Should trigger: 14 of 14 loaded the skill. The guide asks for 90 % at least.
- Should not trigger: 0 of 7 loaded it. The guide asks for none.

Three of the seven off-topic queries loaded another skill instead (a chart,
a classifier, a Markdown conversion): the descriptions share the work as
they should.

| query | autostat loaded | skills called |
|---|---|---|
| What is going on in these eight years of monthly sales? | yes | autostat |
| Find the trend and the seasonality in sales.csv | yes | autostat |
| Is there a yearly cycle in this file? | yes | autostat |
| What is the period of this signal? | yes | autostat |
| Decompose this time series into components | yes | autostat |
| Fit a formula to this noisy curve | yes | autostat |
| Which equation describes these measurements over time? | yes | autostat |
| Write me a PDF report explaining this series | yes | autostat |
| Give me a LaTeX report of this time series | yes | autostat |
| What did the Automatic Statistician find on the airline series? | yes | autostat |
| Run a compositional kernel search on this data | yes | autostat |
| Décompose cette série en tendance et saisonnalité | yes | autostat |
| Quelle formule pour cette courbe bruitée ? | yes | autostat |
| Que contient cette série temporelle ? | yes | autostat |
| What is the weather in Paris? | no | (none) |
| Write a Python function that reverses a string | no | (none) |
| Plot this column as a bar chart | no | dataviz |
| Compare the forecast RMSE of ten models on 500 series | no | (none) |
| Train a gradient-boosted classifier on this table | no | build-ml-pipeline |
| Predict house prices from these twelve features | no | (none) |
| Convert this Markdown file to PDF | no | md2star |
