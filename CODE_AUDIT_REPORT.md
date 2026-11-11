# Code audit report

Project: independent reconstruction of *A Hybrid, Multi-Modal Framework for Fake Account Detection using BERT, GNN and
LightGBM* (package `botrecon`). **Not the original implementation**, which is lost. Audit date: 2026-11-11.

Starting point: `ADHARSH.zip` as uploaded (git commit `371a486`). Every change is a separate commit; `git log -p`
shows the full diff. Reported Kaggle result before the audit: 43 passed, 1 failed
(`test_gcn_learns_and_respects_selection`, 82.5% accuracy against a > 90% threshold). The complete Kaggle log was not
available; the failure was reproduced independently (below).

## Defects found, root causes and fixes

| ID | Severity | Defect | Root cause | Fix | Commit |
| --- | --- | --- | --- | --- | --- |
| D1 | High | `test_gcn_learns_and_respects_selection` failed (0.825 < 0.9); the notebook stops on any failing test | **The test's synthetic data, not the GCN.** A two-cluster kNN graph with only 87.3% same-class edges and 40 test nodes (1 node = 2.5 points). Neighbour averaging pulls boundary nodes to the wrong class, capping the GCN's own training accuracy near 95%. Evidence: an independent NumPy replica of the GCN (same initialisation, AdamW, dropout and selection) scored 0.825–0.975 across seeds (10-seed mean 0.897), matching Kaggle's 0.825; 300 epochs did not help (0.892), so it is not undertraining; logistic regression on the raw features scored 0.975 | Replaced with a stochastic block model where the signal is in the graph (features alone ~0.65–0.70; 94% same-class edges; 200 test nodes). NumPy replica: 0.985–1.000 over 15 runs. The test now also requires the GCN to beat the same model without edges by ≥ 0.15 (replica: ≥ 0.27), checks that validation-loss selection returns a model no worse on validation than the last epoch, checks output shapes, and refuses masked labels. Threshold 0.93 keeps a wide margin. Added seed-reproducibility and gradient-update tests | `64b5e87` |
| D2 | Medium | Two GCN implementations (`gcn.py` inline class and `nn.GCN`) | Historical: reconstruction runner and suite written separately; risk of silent divergence | `gcn.train_gcn` now uses `nn.GCN`; it also returns validation loss of the selected and the last model | `64b5e87` |
| D3 | High | Run plan required TwiBot-20 | TwiBot-20 access must be requested from its authors; Cresci-2017-only runs were impossible meanwhile | TwiBot-20 optional everywhere; finished Cresci-2017 runs are kept when it is added | `9a27547` |
| D4 | High | Required ablations Metadata + GCN and Text + GCN missing | Suite only had GNNs on all node features | `GNN(kind, features)`; `GCN (meta)`, `GCN (text)`; table T3b maps every required combination to a method | `9a27547` |
| D5 | Medium | Macro precision/recall not reported; no programmatic consistency check | Only macro-F1 was recorded | Macro P/R added; `metrics.check_consistency` verifies P, R, F1, accuracy against the confusion counts and that weighted recall equals accuracy; averaging conventions documented in `metrics.py` | `9a27547` |
| D6 | Low | Infinite metadata values not explicitly handled | Current formulas cannot produce inf, but corrupt inputs could | `raw_features` and `MetadataTransformer` map ±inf to missing, imputed from training rows | `9a27547` |
| D7 | Medium | Suite derived "text available" from embedding values | Coupled availability to the encoder | Availability now comes from the text (non-empty) | `9a27547` |
| D8 | Medium | No model checkpoints | Resume worked per run, but trained models were discarded | `save`/`load` for GCN-family models, RGF (incl. temperature), reconstructed hybrids and LightGBM/XGBoost; saved for `checkpoint_seeds` (default 0); round-trip test | `9a27547` |
| D9 | Medium | No reproducibility test for neural code | Not written | Same seed gives identical GCN output; different seed differs | `64b5e87` |
| D10 | Medium | `requirements.txt` lacked `xgboost`, `matplotlib` | Omission | Added | `9a27547` |
| D11 | Medium | Notebook not in the required order; path detection untested inside the notebook; cells depended on earlier variables; no config file | Notebook grew around one `plan` command | 19-step notebook; `config/kaggle_cpu.json`; tested modules `datapaths`, `settings`, `env`, `notebook`, `walkthrough`, `repro_report`; every cell imports what it uses and reads state from disk | `ca83b35`, `ee4f2cc` |
| D12 | Medium | No automated leakage/protocol check on the real data | Guarantees were tested on synthetic data only | `audit.py`: labels, IDs, group sizes, split shares, stratification, scaler fit rows, test-label masking, test isolation in the inductive graph; saved to `results/protocol_audit.json`; any blocking failure stops the notebook | `220f8f1` |
| D13 | Low | No persistent logs | Output only in the notebook | `plan` writes `results/logs/plan_*.log` | `9a27547` |
| D14 | Low | Old Colab notebook could be mistaken for the current one | Kept from an earlier version | Marked LEGACY | `ee4f2cc` |

## Checked and found correct (no change)

- Label masking: every fit reads `y_view` with test labels set to −1; mutation tests show the suite fails if masking is removed.
- Scaler and imputer fitted on training rows only (corrected protocol); fitted on all rows only in the explicitly labelled manuscript-faithful family.
- kNN graph: no self-edges, symmetrised, duplicates removed; GCN normalisation matches the dense formula D^-1/2 (A + I) D^-1/2.
- Inductive graphs: test accounts absent from training; at inference they only receive messages.
- P1 and P3 are separate, named configurations; the 64-d hidden embedding and 2-d probabilities are separate taps.
- Cresci-2017 explicit label map (genuine = 0, all eight automated groups including fake followers = 1); the manuscript's literal rule is kept only as a labelled sensitivity variant.
- TwiBot-20 official train/dev/test files are used; a random split exists only as a labelled manuscript-faithful variant.

## Not changed, documented

- Class imbalance: no reweighting (as in the manuscript); MCC and PR-AUC are reported alongside F1.
- Text variant B's checkpoint (`all-MiniLM-L6-v2`) is a reconstruction choice; the manuscript names none.
- TwiBot-20 tweet order is undocumented; the first 20 tweets in the file are used for variant A.
- Random Forest models are not checkpointed (size).

## Change after the audit: TwiBot-20 replaced by Cresci-2015

TwiBot-20 is available only on request from its authors and has no legitimate Kaggle copy, and the paper is now
drafted from scratch on new experiments. Changes:

- New loader `botrecon/data/cresci2015.py` (groups E13, TFP = human; FSF, INT, TWT = bot; follower/friend edges among
  labelled accounts; dedup log; published-size check).
- Graph choice is driven by whether a dataset ships edges, not by its name; R-GCN runs on Cresci-2015.
- Plan, notebook, config and dataset locator use Cresci-2017 + Cresci-2015; the locator also extracts tar.gz archives,
  skips the `botrecon_*` code/results archives and does not re-extract on every call.
- Cross-dataset tests remove target accounts that also occur in the source dataset (the Cresci-2017 fake followers
  come from the 2015 study) and report the number removed; a target with no bots left is reported by specificity.
- Paper tables no longer contain numbers from the earlier manuscript or TwiBot-20 literature rows. The literal
  reconstruction and the test-set early-stopping diagnostic are off by default (`--with-reconstruction`).
- The TwiBot-20 loader is kept in the library (and its tests) but is not used anywhere in the workflow.

