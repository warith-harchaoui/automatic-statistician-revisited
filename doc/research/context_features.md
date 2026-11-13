# Context-conditioned, interpretable time-series features: a literature review

Written on 11 November 2026 for the `autostat.features` and `autostat.context` prototype.

I opened every reference below on arXiv, a publisher page or an indexing record, except where a flag says otherwise. "Flag" marks a detail I could not confirm on a page I opened.

## 1. Knowledge-guided feature engineering and representation

**Karpatne, Atluri, Faghmous, Steinbach, Banerjee, Ganguly, Shekhar, Samatova and Kumar. "Theory-guided Data Science: A New Paradigm for Scientific Discovery from Data." IEEE TKDE 29(10):2318-2331, 2017. https://arxiv.org/abs/1612.08544**
The paper argues that black-box models which work in commercial settings fail on physical systems and that scientific knowledge should constrain the model, the learning and the outputs. *Borrow:* its framing (knowledge as constraint, as prior or as consistency check) is the vocabulary to use when explaining how autostat uses context. *Measure:* how consistent the features are with stated constraints, alongside accuracy.

**von Rueden et al. (14 authors). "Informed Machine Learning: A Taxonomy and Survey of Integrating Knowledge into Learning Systems." IEEE TKDE, 2021. DOI 10.1109/TKDE.2021.3079836. https://arxiv.org/abs/1903.12394**
It classifies the field along three axes: knowledge source, knowledge representation (equations, logic rules, simulations and so on) and where the knowledge enters the pipeline. *Borrow:* use this taxonomy to give autostat's context a typed schema. Units and positivity are algebraic constraints, expected cycles are priors over the hypothesis set and dated events are regressors or changepoint locations. Each type maps to one injection point: the grammar, the prior or the likelihood.

**Fulcher. "Feature-based time-series analysis." arXiv 2017. https://arxiv.org/abs/1709.08055**
A review of feature-based representations and of large-scale comparison of features across fields. *Borrow:* the argument that features should be interpretable and comparable across datasets. This is the niche autostat's kernel features would occupy.

**Murray, Dervovic and Cashmore. "ELATE: Evolutionary Language model for Automated Time-series Engineering." arXiv preprint, Aug 2025. https://arxiv.org/abs/2508.14667**
An LLM proposes feature transformations that take the domain context into account. Series statistics and feature-importance scores prune the candidates inside an evolutionary loop. The authors report an 8.4% average gain in forecasting accuracy. *Measure against:* it is the closest "context-aware feature generation for time series" baseline. It is not peer reviewed and its features are not structural.

## 2. Interpretable feature baselines and strong non-interpretable baselines

**Lubba, Sethi, Knaute, Schultz, Fulcher and Jones. "catch22: CAnonical Time-series CHaracteristics." https://arxiv.org/abs/1901.10200 (flag: the journal version, which I believe is Data Mining and Knowledge Discovery 2019, was not shown on the page).**
The authors cut hctsa down to 22 features. Across 93 classification datasets, accuracy drops by only about 7% on average while computation is about 1000 times faster. *Use as:* the default cheap interpretable baseline. *Borrow:* the selection method, which keeps features that perform well and are not redundant.

**Fulcher and Jones. "hctsa: A Computational Framework for Automated Time-Series Phenotyping Using Massive Feature Extraction." Cell Systems 5(5):527-531.e3, 2017. Preprint https://arxiv.org/abs/1612.05296**
More than 7,700 features plus tools to pick out the interpretable ones that discriminate. *Use as:* the upper bound on how comprehensive a feature set can be (MATLAB only).

**Christ, Braun, Neuffer and Kempa-Liehr. "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh - A Python package)." Neurocomputing 307:72-77, 2018. DOI 10.1016/j.neucom.2018.03.067. Algorithm: Christ, Kempa-Liehr and Feindt, https://arxiv.org/abs/1610.07717**
63 characterisation methods give 794 default features. A filter based on hypothesis tests controls the share of irrelevant features it keeps. *Borrow:* its FRESH relevance filter, applied to autostat features. *Use as:* a baseline, through FreshPRINCE (see the bake-off below).

**Barandas, Folgado, Fernandes, Santos, Abreu, Bota, Liu, Schultz and Gamboa. "TSFEL: Time Series Feature Extraction Library." SoftwareX 11:100456, 2020. DOI 10.1016/j.softx.2020.100456**
More than 60 statistical, temporal and spectral features. *Use as:* a fast Python baseline.

**Hyndman, Wang and Laptev. "Large-scale unusual time series detection." ICDM Workshops 2015 (flag: I saw only indexing records and snippets, not the full text).** This paper is the origin of the tsfeatures/feasts features: seasonal strength, spikiness, lumpiness and spectral entropy, many of them computed from an STL decomposition. Anomalous series are found as outliers in the space of the first two principal components. **Montero-Manso, Athanasopoulos, Hyndman and Talagala. "FFORMA: Feature-based forecast model averaging." IJF 36(1):86-92, 2020. DOI 10.1016/j.ijforecast.2019.02.011.** It feeds 42 tsfeatures to XGBoost, which learns weights for combining forecasting methods. It came second in M4. *Borrow:* FFORMA is the template for proving that features are useful for forecasting. Replace tsfeatures with autostat features and compare on M4. Hyndman's PCA outlier approach is the template for anomaly detection.

**Henderson and Fulcher. "An Empirical Evaluation of Time-Series Feature Sets." SFE-TSDM workshop at IEEE ICDM 2021. https://arxiv.org/abs/2110.10914**
It compares hctsa, catch22, tsfresh, TSFEL, Kats, tsfeatures and feasts. Per-feature cost spans about three orders of magnitude, from about 0.1 ms for catch22 and TSFEL to about 3 s for tsfeatures. TSFEL and tsfresh are highly redundant (4 principal components explain 90% of TSFEL's variance). hctsa is the most comprehensive. *Measure:* use the same redundancy and overlap analysis to show that autostat features add information the existing sets do not capture. This is the cleanest test of novelty.

**Dempster, Petitjean and Webb. "ROCKET." DMKD 2020. DOI 10.1007/s10618-020-00701-z. https://arxiv.org/abs/1910.13051. Dempster, Schmidt and Webb. "MINIROCKET." KDD 2021. DOI 10.1145/3447548.3467231. https://arxiv.org/abs/2012.08791**
Random convolutional kernels feed a linear classifier. MiniRocket is up to 75 times faster than ROCKET at essentially the same accuracy and runs all 109 UCR datasets in under 10 minutes. *Use as:* the accuracy ceiling for cheap non-interpretable features.

**Middlehurst, Schäfer and Bagnall. "Bake off redux: a review and experimental evaluation of recent time series classification algorithms." DMKD 2024. DOI 10.1007/s10618-024-01022-1. https://arxiv.org/abs/2304.13029**
The benchmark covers 112 UCR datasets plus 30 new ones. Mean accuracies of the feature-based classifiers: FreshPRINCE 0.855, TSFresh 0.799, Catch22 0.795 and Signatures 0.787. Hydra+MultiROCKET and HIVE-COTE 2 are significantly better than everything else (flag: I did not read their exact numbers). *Measure against:* report the same metrics (accuracy, balanced accuracy, AUROC, NLL, F1), use the same 30 resamples and critical-difference diagrams. *Limitation to state honestly:* UCR series are short, z-normalised and carry no timestamps, units or events. Context-conditioned features have no edge there, so UCR can only show that autostat is not worse.

## 3. GP structure discovery as representation

**Lloyd, Duvenaud, Grosse, Tenenbaum and Ghahramani. "Automatic Construction and Natural-Language Description of Nonparametric Regression Models." https://arxiv.org/abs/1402.4304 (2014, before the window; flag: the AAAI 2014 venue was not shown on the page).** This is the ABCD baseline autostat reimplements.

**Janz, Paige, Rainforth, van de Meent and Wood. "Probabilistic structure discovery in time series data." arXiv 2016. https://arxiv.org/abs/1611.06863**
It replaces greedy search with a full posterior over kernel structures, arguing that greedy search is overconfident. *Borrow:* posterior probabilities over structures, not a single best kernel, are the right features.

**Saad, Cusumano-Towner, Schaechtle, Rinard and Mansinghka. "Bayesian Synthesis of Probabilistic Programs for Automatic Data Modeling." PACMPL 3 (POPL), Art. 37, 2019. https://arxiv.org/abs/1907.06249**
It runs Bayesian synthesis over a probabilistic context-free grammar of GP kernels. Explicitly, it estimates "the probability that the underlying data-generating process has key qualitative properties", such as periodicity or a changepoint. *Borrow directly:* this is "structure as features" in its probabilistic form. Features of the form P(periodic component | data) are interpretable and calibrated.

**Saad, Patton, Hoffman, Saurous and Mansinghka. "Sequential Monte Carlo Learning for Time Series Structure Discovery." ICML 2023, PMLR 202:29473-29489. https://arxiv.org/abs/2307.09607**
SMC plus involutive MCMC over symbolic GP structures runs 10 to 100 times faster than earlier methods. Its point and interval forecasts beat statistical and neural baselines on 1,428 econometric series. *Borrow:* the SMC particle weights give the structure posterior autostat needs for features. *Measure on:* its 1,428-series evaluation.

**Malkomes, Schaff and Garnett. "Bayesian optimization for automated model selection." NeurIPS 2016.** https://proceedings.neurips.cc/paper/2016/hash/3bbfdde8842a5c44a0323518eec97cbe-Abstract.html. It defines a "kernel kernel" between models, based on how each model explains the data and runs Bayesian optimization over model space. **Islam, Padhy, Allaire and Arróyave. "The Kernel Manifold: A Geometric Approach to Gaussian Process Model Selection." arXiv Jan 2026. https://arxiv.org/abs/2601.05371** It measures distances between kernels as expected divergences between GP priors and embeds the kernel library with multidimensional scaling for Bayesian optimization. The authors report beating baselines that include LLM-guided search. *Borrow:* these embeddings turn a discrete structure into a fixed-length continuous vector, which downstream classifiers need.

**Simpson, Davies, Lalchand, Vullo, Durrande and Rasmussen. "Kernel Identification Through Transformers (KITT)." NeurIPS 2021. https://arxiv.org/abs/2106.08185**
A transformer trained on data sampled from kernel priors recommends a kernel in a fraction of a second. *Borrow:* amortised structure inference, to make kernel features cheap enough to compute on thousands of series.

**Li, Fox and Goodman. "Automated Statistical Model Discovery with Language Models." ICML 2024. https://arxiv.org/abs/2402.17879**
An LM proposes probabilistic programs and critiques them inside Box's loop. In a restricted kernel DSL it roughly matches the Automatic Statistician and it can improve expert models under natural-language constraints. *Borrow:* use the LM as a proposal distribution conditioned on context, with the Bayesian score kept as the judge.

**Suwandi, Yin, Wang, Li, Chang and Theodoridis. "Adaptive Kernel Design for Bayesian Optimization Is a Piece of CAKE with LLMs." NeurIPS 2025. https://arxiv.org/abs/2509.17998**
LLMs act as crossover and mutation operators on kernels and BAKER ranks the candidates by BIC combined with expected improvement. *Borrow:* LLM mutation operators on autostat's grammar, still scored by BIC.

**Lee, Hyeon-Woo, Ye-Bin, Nam and Oh. "Automated Model Discovery via Multi-modal & Multi-step Pipeline." arXiv Sep 2025. https://arxiv.org/abs/2509.25946**
A vision-language model (AnalyzerVLM) proposes models from plots and EvaluatorVLM judges local fit and trend generalisation both visually and quantitatively. *Borrow cautiously:* a plot-reading critic for the report. It is not peer reviewed (no venue shown).

**Yun, Shin, Song, Lee and Park. "Automated Kernel Discovery Towards Understanding High-dimensional Bayesian Optimization." arXiv May 2026. https://arxiv.org/abs/2605.20249.** An LLM-driven evolutionary search over kernel forms, selected by leave-one-out CRPS. The authors report a mean rank of about 1.2 among 17 methods. It targets high-dimensional Bayesian optimization, not time series. Search results point to ICML or NeurIPS 2026 listings, but I did not confirm either.

## 4. LLMs injecting context and whether context helps

**Williams, Ashok, Marcotte, Zantedeschi, Subramanian, Riachi, Requeima, Lacoste, Rish, Chapados and Drouin. "Context is Key: A Benchmark for Forecasting with Essential Textual Information." ICML 2025. https://arxiv.org/abs/2410.18959**
71 tasks in 7 domains, where the text is needed to solve each task. Context types are intemporal, future, historical, covariate and causal. The metric is RCRPS: CRPS weighted on a region of interest, with a penalty for constraint violations. Llama-3.1-405B with direct prompting improves by 67.1% with context. However, some models ignore the context or get worse with it, they fail badly on some tasks, no method wins on every context type and many LLM methods are Pareto-dominated on cost by Chronos and Lag-Llama. *Measure on:* CiK is the main public benchmark with context. Its context taxonomy maps almost one-to-one onto autostat inputs: intemporal context gives constraints, future context gives dated events, causal and covariate context give regressors.

**Ashok et al. "Beyond Naïve Prompting: Strategies for Improved Zero-shot Context-aided Forecasting with LLMs." NeurIPS 2025 workshop. https://arxiv.org/abs/2508.09904**
The paper tests many strategies on CiK, including ReDP (explicit reasoning), CorDP (the LLM only corrects an existing forecast), IC-DP (worked examples in the prompt) and RouteDP (routing by task difficulty). *Borrow:* CorDP's division of labour, where a statistical model forecasts and the LLM adjusts it with context, matches "the LLM sets priors, the GP fits". I did not read the numbers.

**Hollmann, Müller and Hutter. "Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering." arXiv 2023 (NeurIPS 2023 per common citation; flag: the page shows no venue). https://arxiv.org/abs/2305.03403**
An LLM writes feature code from the dataset description, with a justification for each feature and keeps features only if they pass validation. It improved 11 of 14 datasets, with mean ROC AUC rising from 0.798 to 0.822. *Borrow:* the generate, validate, keep-or-reject loop with written rationales. Apply it to kernel hypotheses ("weekly retail cycle, so propose Per(7 days)") instead of pandas code.

**Jin et al. "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models." ICLR 2024. https://arxiv.org/abs/2310.01728** It uses Prompt-as-Prefix to add context to a frozen LLM. **Tan, Merrill, Gupta, Althoff and Hartvigsen. "Are Language Models Actually Useful for Time Series Forecasting?" NeurIPS 2024 spotlight. https://arxiv.org/abs/2406.16964** Removing the LLM from such methods, or replacing it with an attention layer, does not hurt accuracy and often helps. **Merrill, Tan, Gupta, Hartvigsen and Althoff. "Language Models Still Struggle to Zero-shot Reason about Time Series." arXiv 2024. https://arxiv.org/abs/2404.11757** LLMs are barely above chance at reasoning about what generated a series and relevant context gives only modest forecasting gains. *Lesson:* LLMs add value through the knowledge in the text, not as sequence models. autostat should use the LLM only to turn context into structured priors and constraints and should always ablate with and without context.

**Liu et al. "Time-MMD: Multi-Domain Multimodal Dataset for Time Series Analysis." NeurIPS 2024 Datasets and Benchmarks. https://arxiv.org/abs/2406.08627**
Paired text and numeric series across 9 domains. Adding text reduces MSE by more than 15% overall and by up to 40% in domains with rich text. *Use as:* a second dataset with context.

## 5. Priors from domain knowledge in structure search

**Taylor and Letham. "Forecasting at Scale." The American Statistician 72(1):37-45, 2018. DOI 10.1080/00031305.2017.1380080 (flag: the publisher and PeerJ pages returned 403; details come from RePEc).** Prophet is a modular regression of trend, seasonalities and holidays, whose parameters analysts set with domain knowledge. *Borrow:* the user-facing interface for context, namely holiday tables with windows, named seasonalities with periods and changepoint candidates. *Measure against:* Prophet with the same context is the obvious baseline.

**Brodersen, Gallusser, Koehler, Remy and Scott. "Inferring causal impact using Bayesian structural time-series models." Annals of Applied Statistics 9(1):247-274, 2015. DOI 10.1214/14-AOAS788. https://arxiv.org/abs/1506.00356**
A BSTS model (local trend, seasonality, covariates with time-varying effects) is fitted by MCMC and used as a counterfactual for an intervention. *Borrow:* for a dated event, the counterfactual effect and its credible interval make a strong interpretable feature, such as "promotion lift" with its uncertainty.

**Guimerà, Reichardt, Aguilar-Mogas, Massucci, Miranda, Pallarès and Sales-Pardo. "A Bayesian machine scientist to aid in the solution of challenging scientific problems." Science Advances 6(5):eaav6971, 2020. https://arxiv.org/abs/2004.12157 (flag: verified through indexing pages; the Science page returned 403).** It runs MCMC over closed-form expressions, scored by an approximate marginal posterior. The prior over expressions is learned from a large corpus of real equations. *Borrow:* learn the structure prior from a corpus. For autostat, that means a prior over kernel structures fitted to structures found on reference collections (M4, Monash), conditioned on domain and frequency metadata.

## (a) Five ideas most worth implementing, ranked

1. **Posterior-probability structure features.** Run AutoGP-style SMC (or BIC weights over the beam) and output calibrated features: P(component), plus period, lengthscale, amplitude share, changepoint location and sharpness, each with its uncertainty. The basis is Saad 2019 and 2023 and Janz 2016. It is interpretable, calibrated and absent from every existing feature library.
2. **A typed context schema compiled into grammar and prior.** Expected cycles raise the prior on Per with that period. Positivity or multiplicativity triggers a log-warped likelihood or product structure. Dated events become fixed changepoint or window kernels or indicator regressors. Units fix the scales. The basis is von Rueden's taxonomy, Prophet and CiK's context types. Deterministic, testable and no LLM required.
3. **An optional LLM translator from free text to the schema, plus LLM proposals.** The LLM writes a schema and kernel hypotheses with rationales (CAAFE, Li et al. 2024, CAKE). BIC or the marginal likelihood stays the sole judge, which guards against the failures documented by Tan et al. and in CiK.
4. **Event-effect features.** For each dated event, compute a BSTS or GP counterfactual lift with a credible interval (CausalImpact), plus residual and anomaly scores against the fitted structure.
5. **Fixed-length embeddings of structure** (kernel-kernel or MDS kernel-manifold coordinates), with an amortised KITT-style recogniser for scale. A FRESH-style relevance filter selects among the outputs.

## (b) Evaluation protocol suggested by the literature

- **Context-rich data:** CiK (71 tasks, RCRPS), Time-MMD (9 domains, text), the AutoGP econometric collection (1,428 series), M4 or Monash for forecasting with frequency metadata and retail data with holiday and promotion calendars (flag: I did not open M5 in this review). UCR/UEA (112 + 30 datasets) only as a no-context check that autostat is not worse.
- **Baselines:** catch22, tsfresh/FreshPRINCE, TSFEL and tsfeatures (FFORMA-style for forecasting), MiniRocket and Hydra+MultiROCKET as the non-interpretable ceiling, plus Prophet and BSTS fed the same context. On CiK, also Chronos/Lag-Llama and direct-prompt LLMs.
- **Metrics:** for classification, accuracy, balanced accuracy, AUROC, NLL, F1 and critical-difference ranks over 30 resamples (bake-off redux). For forecasting, MASE/sMAPE, CRPS/RCRPS and interval coverage. For anomalies, PCA-space outliers (Hyndman 2015) against labelled events. For the features themselves, Henderson-Fulcher redundancy and overlap with existing sets, compute time and recovery of planted structure on synthetic data.
- **Mandatory ablations:** no context, true context, perturbed context and misleading context (CiK shows that some models get worse with context). Also LLM translator against hand-written schema.

## (c) Open gaps where autostat could be novel

1. No feature library outputs **compositional structure with posterior uncertainty**. The Henderson-Fulcher overlap analysis would show whether these features carry new information.
2. Context-aided work is almost all **forecasting by LLMs** (CiK, Time-MMD). Nobody has shown context-conditioned **interpretable features** helping classification or anomaly detection.
3. LLM kernel search (Li 2024, CAKE 2025, Kernel Discovery 2026) conditions on the data, but **none takes business metadata (calendars, units, constraints) as typed input** and none reports a context ablation.
4. **Corpus-learned structure priors** in the style of the Bayesian machine scientist have not been applied to GP kernel grammars.
5. **Constraint violation** (positivity, cycle consistency) is scored in CiK's RCRPS but has never been used as a check on feature quality. autostat could report it for every feature.
