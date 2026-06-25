# Independent submission verification

Lens: JD text + EDA (`eda.py`/`eda2.py`). No `ranker/` import, no re-scoring. Snapshot 2026-06-09. Coherence backbone = summary archetype (verified 21/150/1000), NOT the negation-blind phrase grade.

- Pool N=100,000  archetypes: {'OTHER': 98821, 'STRONG': 150, 'GENERIC': 1000, 'ELITE': 21, 'SENIOR_ENG': 8}
- Submission rows: 100

## 1. Inclusion — hard JD gates among the 100

Pool-wide column confirms each gate is live. cv_primary/pure_research read 0 pool-wide: this dataset encodes domain via narrative (handled by the ranker's domain gate), not via a separable CV/research-only industry cohort.

| gate | pool-wide | trips in 100 | ids |
|---|---|---|---|
| honeypot | 25 | 0 | — |
| services_only | 9745 | 0 | — |
| kw_stuffer | 4570 | 0 | — |
| framework_dabbler | 30517 | 0 | — |
| cv_primary | 0 | 0 | — |
| pure_research | 0 | 0 | — |

## 2. Archetype + evidence of the 100 (independent recompute)

- top-10:  ELITE=7  STRONG=3  SENIOR_ENG=0  GENERIC=0  OTHER=0
- top-50:  ELITE=12  STRONG=34  SENIOR_ENG=2  GENERIC=2  OTHER=0
- top-100: ELITE=13  STRONG=64  SENIOR_ENG=5  GENERIC=18  OTHER=0
- evidence grade (top-100): STRONG=96  MED=4  WEAK=0  NONE=0
- coherent builders (ELITE/STRONG arch) in top-100: 82 | GENERIC/OTHER in top-100: 18

## 3. ELITE builders (21) — reachability & placement

- reachable 15/21 | in-submission 13/21 | excluded-but-reachable 2 | trap-buried 6

| id | arch | rank | reach | active_d | rr | otw | notice | yoe | in_band | in_target | loc_ok | summ_grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAND_0046525 | ELITE | 1 | True | 17 | 0.88 | True | 60 | 6.1 | True | True | True | STRONG |
| CAND_0011687 | ELITE | 2 | True | 30 | 0.89 | True | 15 | 7.8 | True | False | False | STRONG |
| CAND_0018499 | ELITE | 4 | True | 27 | 0.61 | True | 15 | 7.2 | True | True | True | STRONG |
| CAND_0077337 | ELITE | 5 | True | 14 | 0.95 | True | 60 | 7.0 | True | False | True | STRONG |
| CAND_0002025 | ELITE | 6 | True | 14 | 0.8 | True | 30 | 5.9 | True | False | False | STRONG |
| CAND_0046064 | ELITE | 7 | True | 44 | 0.78 | True | 30 | 8.9 | True | False | False | STRONG |
| CAND_0088025 | ELITE | 8 | True | 26 | 0.83 | True | 90 | 8.6 | True | False | False | STRONG |
| CAND_0081846 | ELITE | 11 | True | 37 | 0.73 | True | 30 | 6.7 | True | False | True | STRONG |
| CAND_0071974 | ELITE | 13 | True | 54 | 0.76 | True | 45 | 7.8 | True | False | False | STRONG |
| CAND_0086022 | ELITE | 14 | True | 55 | 0.55 | True | 0 | 5.3 | True | False | True | STRONG |
| CAND_0055905 | ELITE | 20 | True | 23 | 0.87 | True | 30 | 8.1 | True | False | False | STRONG |
| CAND_0008425 | ELITE | 32 | True | 45 | 0.66 | True | 90 | 7.8 | True | False | False | STRONG |
| CAND_0005260 | ELITE | 83 | True | 30 | 0.86 | False | 60 | 5.2 | True | False | True | STRONG |
| CAND_0007411 | ELITE | — | False | 189 | 0.12 | False | 15 | 8.0 | True | False | True | STRONG |
| CAND_0033861 | ELITE | — | False | 113 | 0.16 | False | 30 | 8.0 | True | False | False | STRONG |
| CAND_0039754 | ELITE | — | True | 26 | 0.81 | True | 30 | 16.2 | False | False | True | STRONG |
| CAND_0041611 | ELITE | — | False | 188 | 0.07 | False | 30 | 6.4 | True | False | False | STRONG |
| CAND_0060072 | ELITE | — | False | 136 | 0.1 | False | 90 | 5.7 | True | False | True | STRONG |
| CAND_0092278 | ELITE | — | False | 214 | 0.07 | False | 90 | 6.8 | True | True | True | STRONG |
| CAND_0093547 | ELITE | — | True | 60 | 0.75 | False | 60 | 2.9 | False | False | False | STRONG |
| CAND_0094759 | ELITE | — | False | 152 | 0.11 | False | 30 | 8.6 | True | True | True | STRONG |

## 4. Coherent builders (171) — exclusion accounting

- in top-100: 82 | excluded: 97
- excluded by reason: {'disqualified': 5, 'CLEAN-MISS': 37, 'out-of-location': 41, 'unreachable': 6, 'out-of-band(yoe)': 8}
- **CLEAN-MISS** (reachable+in-band+located+clean yet excluded): 37
- coherent-FIT pool: 92 | in top-100: 55 | excluded: 37

### Excluded coherent-FIT (the candidates a correct ranker must justify dropping)

| id | arch | rank | reach | active_d | rr | otw | notice | yoe | in_band | in_target | loc_ok | summ_grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAND_0003977 | STRONG | — | True | 83 | 0.55 | False | 45 | 4.6 | True | True | True | STRONG |
| CAND_0006557 | STRONG | — | True | 60 | 0.63 | True | 120 | 7.9 | True | False | True | STRONG |
| CAND_0007009 | STRONG | — | True | 82 | 0.62 | True | 30 | 7.9 | True | True | True | STRONG |
| CAND_0007460 | STRONG | — | True | 19 | 0.48 | True | 120 | 4.7 | True | True | True | STRONG |
| CAND_0008239 | STRONG | — | True | 86 | 0.73 | True | 15 | 4.0 | True | False | True | STRONG |
| CAND_0009837 | STRONG | — | True | 70 | 0.53 | False | 60 | 4.3 | True | True | True | STRONG |
| CAND_0012957 | STRONG | — | True | 38 | 0.67 | False | 120 | 4.9 | True | False | True | STRONG |
| CAND_0015578 | STRONG | — | True | 69 | 0.65 | True | 90 | 5.4 | True | True | True | STRONG |
| CAND_0020877 | STRONG | — | True | 67 | 0.66 | False | 60 | 5.1 | True | True | True | STRONG |
| CAND_0022812 | STRONG | — | True | 82 | 0.79 | False | 60 | 4.5 | True | False | True | STRONG |
| CAND_0026532 | STRONG | — | True | 73 | 0.52 | True | 90 | 4.8 | True | False | True | STRONG |
| CAND_0029367 | STRONG | — | True | 20 | 0.77 | True | 90 | 5.7 | True | True | True | STRONG |
| CAND_0030348 | STRONG | — | True | 71 | 0.54 | True | 45 | 4.5 | True | True | True | STRONG |
| CAND_0031593 | STRONG | — | True | 82 | 0.58 | True | 90 | 7.8 | True | False | True | STRONG |
| CAND_0036437 | STRONG | — | True | 26 | 0.87 | False | 30 | 4.8 | True | False | True | STRONG |
| CAND_0036863 | STRONG | — | True | 55 | 0.46 | True | 60 | 4.3 | True | False | True | STRONG |
| CAND_0044855 | STRONG | — | True | 46 | 0.57 | False | 60 | 6.6 | True | False | True | STRONG |
| CAND_0045250 | STRONG | — | True | 41 | 0.74 | False | 15 | 6.6 | True | True | True | STRONG |
| CAND_0047721 | STRONG | — | True | 65 | 0.49 | False | 90 | 7.0 | True | False | True | STRONG |
| CAND_0054123 | STRONG | — | True | 32 | 0.87 | False | 60 | 4.7 | True | True | True | STRONG |
| CAND_0057563 | STRONG | — | True | 49 | 0.83 | False | 60 | 6.8 | True | False | True | STRONG |
| CAND_0058575 | STRONG | — | True | 65 | 0.69 | True | 90 | 5.8 | True | False | True | STRONG |
| CAND_0060054 | STRONG | — | True | 84 | 0.86 | False | 15 | 6.4 | True | False | True | STRONG |
| CAND_0061339 | STRONG | — | True | 64 | 0.9 | True | 90 | 4.2 | True | False | True | STRONG |
| CAND_0064270 | STRONG | — | True | 81 | 0.82 | False | 45 | 4.2 | True | True | True | STRONG |
| CAND_0066376 | STRONG | — | True | 37 | 0.51 | True | 90 | 5.7 | True | True | True | STRONG |
| CAND_0068351 | SENIOR_ENG | — | True | 59 | 0.86 | False | 0 | 6.4 | True | True | True | STRONG |
| CAND_0076251 | STRONG | — | True | 55 | 0.51 | True | 60 | 7.6 | True | False | True | STRONG |
| CAND_0079284 | STRONG | — | True | 59 | 0.79 | False | 30 | 4.9 | True | True | True | STRONG |
| CAND_0080766 | SENIOR_ENG | — | True | 61 | 0.66 | False | 0 | 8.8 | True | False | True | STRONG |
| CAND_0086151 | STRONG | — | True | 78 | 0.52 | True | 120 | 7.7 | True | False | True | STRONG |
| CAND_0087364 | STRONG | — | True | 40 | 0.48 | False | 30 | 4.9 | True | True | True | STRONG |
| CAND_0087630 | STRONG | — | True | 85 | 0.45 | False | 30 | 7.2 | True | True | True | STRONG |
| CAND_0089546 | STRONG | — | True | 66 | 0.64 | False | 90 | 4.8 | True | True | True | STRONG |
| CAND_0089552 | STRONG | — | True | 43 | 0.49 | False | 120 | 6.0 | True | False | True | STRONG |
| CAND_0094056 | STRONG | — | True | 54 | 0.82 | False | 120 | 5.9 | True | False | True | STRONG |
| CAND_0099401 | STRONG | — | True | 13 | 0.42 | False | 90 | 7.7 | True | False | True | STRONG |

### All other excluded coherent builders (behaviorally/structurally explained)

| id | arch | rank | reach | active_d | rr | otw | notice | yoe | in_band | in_target | loc_ok | summ_grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAND_0001610 | STRONG | — | True | 58 | 0.57 | True | 90 | 3.0 | False | False | True | STRONG |
| CAND_0019480 | STRONG | — | True | 27 | 0.87 | True | 90 | 2.8 | False | False | True | STRONG |
| CAND_0037000 | STRONG | — | True | 24 | 0.61 | False | 15 | 2.7 | False | True | True | STRONG |
| CAND_0039521 | STRONG | — | True | 59 | 0.41 | False | 30 | 3.0 | False | False | False | STRONG |
| CAND_0093547 | ELITE | — | True | 60 | 0.75 | False | 60 | 2.9 | False | False | False | STRONG |
| CAND_0010770 | STRONG | — | True | 35 | 0.73 | True | 30 | 15.2 | False | False | True | STRONG |
| CAND_0013536 | STRONG | — | True | 42 | 0.87 | True | 90 | 14.1 | False | False | False | STRONG |
| CAND_0039754 | ELITE | — | True | 26 | 0.81 | True | 30 | 16.2 | False | False | True | STRONG |
| CAND_0055992 | STRONG | — | True | 73 | 0.72 | True | 60 | 16.9 | False | False | False | STRONG |
| CAND_0071115 | STRONG | — | True | 57 | 0.5 | True | 90 | 16.5 | False | False | True | STRONG |
| CAND_0091534 | STRONG | — | True | 35 | 0.84 | False | 30 | 16.6 | False | True | True | STRONG |
| CAND_0093331 | STRONG | — | True | 62 | 0.41 | False | 30 | 16.1 | False | False | True | STRONG |
| CAND_0095619 | STRONG | — | True | 54 | 0.9 | True | 30 | 15.6 | False | True | True | STRONG |
| CAND_0005538 | SENIOR_ENG | — | True | 23 | 0.81 | True | 90 | 5.9 | True | False | False | STRONG |
| CAND_0011432 | STRONG | — | True | 85 | 0.67 | True | 60 | 7.6 | True | False | False | STRONG |
| CAND_0013613 | STRONG | — | True | 78 | 0.73 | True | 60 | 4.7 | True | False | False | STRONG |
| CAND_0015528 | STRONG | — | True | 54 | 0.53 | False | 30 | 7.4 | True | False | False | STRONG |
| CAND_0020350 | STRONG | — | True | 66 | 0.58 | True | 30 | 5.8 | True | False | False | STRONG |
| CAND_0020708 | STRONG | — | True | 76 | 0.84 | True | 30 | 4.2 | True | False | False | STRONG |
| CAND_0024466 | STRONG | — | True | 18 | 0.47 | True | 120 | 5.2 | True | False | False | STRONG |
| CAND_0024620 | STRONG | — | True | 58 | 0.41 | True | 45 | 5.9 | True | False | False | STRONG |
| CAND_0030031 | STRONG | — | True | 18 | 0.94 | False | 30 | 5.7 | True | False | False | STRONG |
| CAND_0030827 | STRONG | — | True | 61 | 0.52 | True | 120 | 5.4 | True | False | False | STRONG |
| CAND_0030953 | STRONG | — | True | 59 | 0.63 | False | 45 | 7.8 | True | False | False | STRONG |
| CAND_0032515 | STRONG | — | True | 26 | 0.54 | False | 45 | 5.1 | True | False | False | STRONG |
| CAND_0032807 | STRONG | — | True | 79 | 0.51 | True | 90 | 4.2 | True | False | False | STRONG |
| CAND_0037944 | STRONG | — | True | 79 | 0.42 | True | 30 | 4.9 | True | False | False | STRONG |
| CAND_0040117 | STRONG | — | True | 84 | 0.66 | True | 15 | 6.5 | True | False | False | STRONG |
| CAND_0040887 | STRONG | — | True | 16 | 0.84 | True | 15 | 4.7 | True | False | False | STRONG |
| CAND_0041568 | STRONG | — | True | 64 | 0.52 | True | 90 | 5.2 | True | False | False | STRONG |
| CAND_0043228 | STRONG | — | True | 18 | 0.41 | False | 30 | 6.8 | True | False | False | STRONG |
| CAND_0044883 | STRONG | — | True | 33 | 0.84 | False | 90 | 6.3 | True | False | False | STRONG |
| CAND_0049538 | STRONG | — | True | 13 | 0.72 | False | 30 | 5.8 | True | False | False | STRONG |
| CAND_0050876 | STRONG | — | True | 14 | 0.67 | True | 90 | 6.0 | True | False | False | STRONG |
| CAND_0051292 | STRONG | — | True | 58 | 0.52 | True | 30 | 5.2 | True | False | False | STRONG |
| CAND_0051615 | STRONG | — | True | 88 | 0.88 | True | 60 | 4.6 | True | False | False | STRONG |
| CAND_0051630 | STRONG | — | True | 66 | 0.51 | False | 90 | 6.0 | True | False | False | STRONG |
| CAND_0056881 | STRONG | — | True | 44 | 0.75 | False | 90 | 4.5 | True | False | False | STRONG |
| CAND_0057701 | STRONG | — | True | 30 | 0.56 | True | 120 | 4.1 | True | False | False | STRONG |
| CAND_0061655 | STRONG | — | True | 19 | 0.88 | False | 15 | 4.6 | True | False | False | STRONG |
| CAND_0065878 | STRONG | — | True | 78 | 0.48 | True | 15 | 7.8 | True | False | False | STRONG |
| CAND_0070202 | STRONG | — | True | 20 | 0.6 | False | 90 | 5.1 | True | False | False | STRONG |
| CAND_0070485 | STRONG | — | True | 68 | 0.45 | True | 120 | 6.4 | True | False | False | STRONG |
| CAND_0074735 | STRONG | — | True | 73 | 0.77 | True | 90 | 5.5 | True | False | False | STRONG |
| CAND_0075439 | STRONG | — | True | 33 | 0.56 | False | 30 | 4.3 | True | False | False | STRONG |
| CAND_0076904 | STRONG | — | True | 49 | 0.54 | True | 90 | 4.2 | True | False | False | STRONG |
| CAND_0077285 | STRONG | — | True | 37 | 0.56 | True | 60 | 5.5 | True | False | False | STRONG |
| CAND_0078042 | STRONG | — | True | 46 | 0.91 | False | 30 | 4.7 | True | False | False | STRONG |
| CAND_0081686 | STRONG | — | True | 45 | 0.91 | False | 60 | 6.0 | True | False | False | STRONG |
| CAND_0083307 | STRONG | — | True | 80 | 0.7 | True | 120 | 7.8 | True | False | False | STRONG |
| CAND_0084819 | STRONG | — | True | 31 | 0.74 | True | 120 | 4.5 | True | False | False | STRONG |
| CAND_0092245 | STRONG | — | True | 73 | 0.5 | True | 60 | 4.1 | True | False | False | STRONG |
| CAND_0095528 | STRONG | — | True | 37 | 0.55 | True | 45 | 5.3 | True | False | False | STRONG |
| CAND_0096172 | STRONG | — | True | 37 | 0.47 | False | 45 | 5.2 | True | False | False | STRONG |
| CAND_0007411 | ELITE | — | False | 189 | 0.12 | False | 15 | 8.0 | True | False | True | STRONG |
| CAND_0033861 | ELITE | — | False | 113 | 0.16 | False | 30 | 8.0 | True | False | False | STRONG |
| CAND_0041611 | ELITE | — | False | 188 | 0.07 | False | 30 | 6.4 | True | False | False | STRONG |
| CAND_0060072 | ELITE | — | False | 136 | 0.1 | False | 90 | 5.7 | True | False | True | STRONG |
| CAND_0092278 | ELITE | — | False | 214 | 0.07 | False | 90 | 6.8 | True | True | True | STRONG |
| CAND_0094759 | ELITE | — | False | 152 | 0.11 | False | 30 | 8.6 | True | True | True | STRONG |

## 5. GENERIC/OTHER occupying top-100 slots (weaker includes)

These hold slots a reviewer could challenge; legitimate only if no excluded coherent-FIT dominates them on JD axes.

| id | arch | rank | reach | active_d | rr | otw | notice | yoe | in_band | in_target | loc_ok | summ_grade |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CAND_0068932 | GENERIC | 42 | True | 14 | 0.82 | True | 30 | 5.2 | True | True | True | STRONG |
| CAND_0073314 | GENERIC | 50 | True | 15 | 0.84 | True | 60 | 5.2 | True | True | True | MED |
| CAND_0008295 | GENERIC | 59 | True | 14 | 0.89 | True | 45 | 6.5 | True | True | True | STRONG |
| CAND_0067866 | GENERIC | 60 | True | 18 | 0.79 | True | 45 | 6.4 | True | True | True | MED |
| CAND_0070525 | GENERIC | 61 | True | 15 | 0.93 | True | 45 | 5.4 | True | True | True | MED |
| CAND_0092706 | GENERIC | 66 | True | 13 | 0.78 | True | 120 | 5.8 | True | True | True | STRONG |
| CAND_0070514 | GENERIC | 67 | True | 20 | 0.8 | True | 60 | 5.5 | True | True | True | MED |
| CAND_0048558 | GENERIC | 73 | True | 13 | 0.8 | True | 30 | 6.7 | True | True | True | MED |
| CAND_0010603 | GENERIC | 74 | True | 17 | 0.94 | True | 90 | 5.3 | True | True | True | STRONG |
| CAND_0072688 | GENERIC | 75 | True | 17 | 0.87 | True | 45 | 6.9 | True | True | True | STRONG |
| CAND_0024549 | GENERIC | 78 | True | 16 | 0.85 | True | 45 | 5.2 | True | True | True | STRONG |
| CAND_0004402 | GENERIC | 79 | True | 15 | 0.83 | True | 60 | 6.0 | True | True | True | MED |
| CAND_0053605 | GENERIC | 86 | True | 18 | 0.88 | True | 30 | 6.9 | True | True | True | MED |
| CAND_0073007 | GENERIC | 88 | True | 33 | 0.83 | True | 60 | 5.8 | True | True | True | MED |
| CAND_0010149 | GENERIC | 89 | True | 15 | 0.82 | True | 90 | 6.9 | True | True | True | MED |
| CAND_0082086 | GENERIC | 91 | True | 26 | 0.85 | True | 45 | 6.0 | True | True | True | MED |
| CAND_0083852 | GENERIC | 92 | True | 13 | 0.85 | True | 90 | 6.0 | True | False | True | STRONG |
| CAND_0066690 | GENERIC | 93 | True | 13 | 0.88 | True | 30 | 4.8 | True | True | True | MED |

## 6. Boilerplate-divergence (expected, NOT misses)

- non-archetype candidates scored phrase-STRONG only via shuffled career descriptions: excluded=530, in-submission=14.
  These are the negation-blind / boilerplate false-positives the ranker deliberately discounts (S1/S2). Their exclusion is correct, not a miss.

## 7. Ordering sanity

- mean independent strength: top10=4.35  11-50=3.74  51-100=3.33
- Spearman(submission_rank, independent_strength) = 0.753

## 8. Head-to-head — unjustified swaps (JD-axis dominance)

Excluded coherent-FIT builder that is >= every included GENERIC/OTHER on rr, recency, open-to-work, notice, seniority, location AND stability, while a higher tier. Recency/notice use small operational slack; seniority/stability do not. Zero => all swaps are legitimate availability/context trade-offs.

- unjustified swap pairs: 0 | distinct excluded builders: 0 | distinct included slots dominated: 0

- cohort availability (the JD's down-weight lever):
  - excluded coherent-FIT: rr=0.65 active=58d otw=41% notice=66d in_target=46%
  - included GENERIC/OTHER: rr=0.85 active=17d otw=100% notice=57d in_target=94%
