### TABLE 1. Dataset statistics

| Property | Cresci-2017 | Cresci-2015 |
|---|---|---|
| Accounts (after deduplication) | 14,368 | 5,301 |
| Bots / humans | 10,894 / 3,474 | 3,351 / 1,950 |
| Bot share (%) | 75.8 | 63.2 |
| Human groups | genuine accounts | E13, TFP |
| Bot groups | social spambots 1-3, traditional spambots 1-4, fake followers | FSF, INT, TWT |
| Empty profile description, bots / humans (%) | 49.2 / 10.9 | 32.0 / 7.4 |
| Graph | none released; kNN (k = 10, cosine) on metadata | follower + friend relations |
| Directed edges among labelled accounts | — | 7,108 |
| Isolated accounts (all / test, %) | 0 / 0.0 | 3,381 / 64.3 |
| Test accounts without text (%) | 40.6 | 21.0 |
| Edge homophily, training nodes | 0.894 | 0.973 |

*Counts from the run's data audit (data_audit.json). Cresci-2017 group traditional spambots 3 has 403 of the 433 published accounts. The Cresci-2017 kNN graph is built from the same metadata the models use.*
