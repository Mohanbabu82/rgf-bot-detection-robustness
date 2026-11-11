# Test report

## Where the tests were run

- **This report:** Linux, Python 3.13.18, 2 CPU cores, **no PyTorch / LightGBM / XGBoost / transformers and no network**. Command: `python -m pytest -q -rs -p no:cacheprovider`.
- **Result here (after the Cresci-2015 change): 72 passed, 0 failed, 0 errors, 15 skipped.** The skipped tests need PyTorch, LightGBM or XGBoost; they were **not executed** and are not claimed to pass. They run in notebook step 2 on Kaggle, which stops if any test fails.
- **Before the audit (reported from Kaggle, log not available):** 43 passed, 1 failed (`test_gcn_learns_and_respects_selection`, 0.825 vs > 0.9).

## The failed test: diagnosis

An independent NumPy replica of the GCN (same initialisation, AdamW, dropout 0.5, validation-loss selection) was run on the test's own data:

| Configuration (original test data, 10 seeds) | Mean test accuracy | Minimum | Mean train accuracy |
| --- | --- | --- | --- |
| Manuscript settings (100 epochs, lr 0.01) | 0.897 | 0.825 | 0.947 |
| 300 epochs | 0.892 | 0.850 | 0.951 |
| lr 0.05 | 0.912 | 0.825 | 0.948 |
| No dropout | 0.908 | 0.825 | 0.942 |
| No model selection (last epoch) | 0.783 | 0.750 | 0.957 |
| Reference: logistic regression on raw features | 0.975 | | |

Edge homophily of that kNN graph: 0.873; test set: 40 nodes. Conclusion: the data and threshold, not the model or the training budget. On the replacement stochastic-block-model data (homophily ~0.94, 200 test nodes) the same replica scored 0.985–1.000 over 15 runs, the no-edge model at most 0.715, and feature-only logistic regression at most 0.705. The real PyTorch run of the new test happens on Kaggle.

## Every test

| Area | File | Test | Status here | Note / fix applied |
| --- | --- | --- | --- | --- |
| 14 end-to-end CLI (stubs) | test_cli.py | `test_cli_smoke_end_to_end` | passed |  |
| 14 end-to-end CLI (stubs) | test_cli.py | `test_resume_and_time_budget` | passed |  |
| 1-2 dataset loading, labels | test_data.py | `test_explicit_map_counts_fake_followers_as_bots` | passed |  |
| 1-2 dataset loading, labels | test_data.py | `test_literal_rule_reproduces_algorithm1_flaw` | passed |  |
| 1-2 dataset loading, labels | test_data.py | `test_cresci_dedup_and_labels` | passed |  |
| 1-2 dataset loading, labels | test_data.py | `test_cresci_metadata_parsed` | passed |  |
| 1-2 dataset loading, labels | test_data.py | `test_cresci_tweets_recent_first_and_capped` | passed |  |
| 1-2 dataset loading, labels | test_data.py | `test_twibot_loader` | passed |  |
| 1-2 dataset loading, labels | test_data.py | `test_cresci2015_loader` | passed | NEW (Cresci-2015 dataset) |
| 1-2 dataset loading, labels | test_data.py | `test_cresci2015_loader_errors` | passed | NEW (Cresci-2015 dataset) |
| environment, dataset paths, config | test_env_paths_config.py | `test_environment_report_and_missing_package_error` | passed | NEW (D11) |
| environment, dataset paths, config | test_env_paths_config.py | `test_locate_plain_folders` | passed | NEW (D11); updated: TwiBot-20 replaced by Cresci-2015 |
| environment, dataset paths, config | test_env_paths_config.py | `test_locate_cresci2015_targz_with_inner_zips` | passed | NEW (Cresci-2015 dataset) |
| environment, dataset paths, config | test_env_paths_config.py | `test_locate_nested_zips_like_the_official_release` | passed | NEW (D11); updated: TwiBot-20 replaced by Cresci-2015 |
| environment, dataset paths, config | test_env_paths_config.py | `test_locate_explicit_paths_and_errors` | passed | NEW (D11); updated: TwiBot-20 replaced by Cresci-2015 |
| environment, dataset paths, config | test_env_paths_config.py | `test_config_defaults_and_validation` | passed | NEW (D11); updated: TwiBot-20 replaced by Cresci-2015 |
| environment, dataset paths, config | test_env_paths_config.py | `test_repository_config_file_is_valid` | passed | NEW (D11) |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_username_entropy` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_raw_feature_formulas` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_scaler_fitted_only_on_fit_rows` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_splits_disjoint_stratified_deterministic` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_knn_transductive` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_knn_inductive_has_no_test_leakage_paths` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_follow_inductive` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_normalised_adjacency_matches_dense_formula` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_homophily_reads_only_given_nodes` | passed |  |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_infinite_values_are_imputed_from_training_rows` | passed | NEW (D6) |
| 3, 6, 9 metadata, graph, splits | test_features_graph_splits.py | `test_raw_features_never_infinite` | passed | NEW (D6) |
| 11 metrics, statistics | test_metrics_stats.py | `test_bot_class_metrics_by_hand` | passed |  |
| 11 metrics, statistics | test_metrics_stats.py | `test_weighted_recall_equals_accuracy` | passed |  |
| 11 metrics, statistics | test_metrics_stats.py | `test_mean_ci_uses_t_distribution` | passed |  |
| 11 metrics, statistics | test_metrics_stats.py | `test_paired_and_mcnemar_and_bootstrap_and_holm` | passed |  |
| 11 metrics, statistics | test_metrics_stats.py | `test_identical_results_are_not_significant` | passed |  |
| 11 metrics, statistics | test_metrics_stats.py | `test_metric_definitions_are_consistent` | passed | NEW (D5) |
| notebook steps 8-10, 19 | test_notebook_helpers.py | `test_ctx_and_budget` | passed | NEW (D11); updated: TwiBot-20 replaced by Cresci-2015 |
| notebook steps 8-10, 19 | test_notebook_helpers.py | `test_walkthrough_steps_without_torch[True]` | passed | NEW (D11); updated: TwiBot-20 replaced by Cresci-2015 |
| notebook steps 8-10, 19 | test_notebook_helpers.py | `test_walkthrough_steps_without_torch[False]` | passed | NEW (D11); updated: TwiBot-20 replaced by Cresci-2015 |
| notebook steps 8-10, 19 | test_notebook_helpers.py | `test_reproducibility_record_lists_pending_runs` | passed | NEW (D11); updated: TwiBot-20 replaced by Cresci-2015 |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_corrected_blend[P3-hidden-twibot20]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_corrected_blend[P3-hidden-cresci2017]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_corrected_blend[P3-prob-twibot20]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_corrected_blend[P3-prob-cresci2017]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_corrected_blend[P1-hidden-twibot20]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_corrected_blend[P1-hidden-cresci2017]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_corrected_blend[P1-prob-twibot20]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_corrected_blend[P1-prob-cresci2017]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_literal_insample_and_scaler_all` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_oof_requires_prob_tap` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_oof_runs_with_prob_tap` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_inductive_training_graph_excludes_test[twibot20]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_inductive_training_graph_excludes_test[cresci2017]` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_diagnostic_is_labelled_invalid` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_provenance_labels` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_grid_sizes` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_empty_text_rates` | passed |  |
| 9-10 isolation, LightGBM inputs (stubs) | test_pipeline.py | `test_report_builds_and_keeps_provenance_apart` | passed |  |
| run plan | test_plan.py | `test_closest_literal` | passed | updated: TwiBot-20 replaced by Cresci-2015 |
| run plan | test_plan.py | `test_plan_order_and_budget` | passed | UPDATED: parts main/extended, log file; updated: TwiBot-20 replaced by Cresci-2015 |
| run plan | test_plan.py | `test_plan_cresci_only` | passed | NEW (D3) |
| run plan | test_plan.py | `test_plan_passes_the_partner_dataset_for_cross_tests` | passed | NEW (Cresci-2015 dataset) |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gcn_forward_matches_dense_formula` | skipped (needs torch; runs on Kaggle) |  |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gcn_learns_and_respects_selection[0]` | skipped (needs torch; runs on Kaggle) | REPLACED (D1): graph-dependent SBM data, no-edge control, selection check |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gcn_learns_and_respects_selection[1]` | skipped (needs torch; runs on Kaggle) | REPLACED (D1): graph-dependent SBM data, no-edge control, selection check |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gcn_seed_reproducibility` | skipped (needs torch; runs on Kaggle) | FAILED on Kaggle (1 of 87): test helper gave an empty validation set at n = 300; helper now splits 50/17/33 at any n (n = 600 unchanged) |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gcn_gradients_update_every_parameter` | skipped (needs torch; runs on Kaggle) | NEW |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_lightgbm_stage` | skipped (needs lightgbm; runs on Kaggle) |  |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gnn_baselines_learn[gcn]` | skipped (needs torch; runs on Kaggle) |  |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gnn_baselines_learn[sage]` | skipped (needs torch; runs on Kaggle) |  |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gnn_baselines_learn[gat]` | skipped (needs torch; runs on Kaggle) |  |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_gnn_baselines_learn[rgcn]` | skipped (needs torch; runs on Kaggle) |  |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_rgf_gating_and_masks` | skipped (needs torch; runs on Kaggle) |  |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_every_method_end_to_end_on_synthetic` | skipped (needs torch; runs on Kaggle) | updated: TwiBot-20 replaced by Cresci-2015 |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_checkpoint_roundtrip_reproduces_predictions` | skipped (needs torch; runs on Kaggle) | NEW (D8) |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_feature_subset_gcn_uses_only_its_features` | skipped (needs torch; runs on Kaggle) | NEW (D4) |
| 4-8, 10, 12-14 real PyTorch / LightGBM / XGBoost | test_real_backends.py | `test_walkthrough_gcn_lightgbm_evaluation` | skipped (needs torch; runs on Kaggle) | NEW (notebook steps 11-14); updated: TwiBot-20 replaced by Cresci-2015 |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_perturbations_touch_only_test_nodes` | passed |  |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_calibration_metrics` | passed |  |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_suite_end_to_end_and_report` | passed | updated: TwiBot-20 replaced by Cresci-2015 |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_suite_time_budget` | passed |  |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_inductive_prepared_hides_test_nodes` | passed |  |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_checkpoints_saved_only_for_requested_seeds` | passed | NEW (D8) |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_modality_table_rows` | passed | NEW (D4) |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_protocol_audit_passes_on_synthetic_releases[cresci]` | passed | NEW (D12) |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_protocol_audit_passes_on_synthetic_releases[cresci2015]` | passed | NEW (Cresci-2015 dataset) |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_protocol_audit_passes_on_synthetic_releases[twibot]` | passed | NEW (D12) |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_protocol_audit_detects_literal_label_rule` | passed | NEW (D12) |
| 9, 13-14 suite, protocol audit, checkpoints (stubs) | test_suite.py | `test_cross_dataset_removes_shared_accounts` | passed | NEW (Cresci-2015 dataset) |

## Coverage of the requested test areas

| # | Area | Tests | Executed here |
| --- | --- | --- | --- |
| 1 | Dataset loading (Cresci-2017, Cresci-2015) | test_data, test_env_paths_config | yes |
| 2 | Label mapping | test_data (explicit and literal rules), test_protocol_audit_* | yes |
| 3 | Metadata processing | test_raw_feature_formulas, test_scaler_fitted_only_on_fit_rows, test_infinite_* | yes |
| 4 | Text processing | test_walkthrough_steps_without_torch (hashing test encoder), empty-text tests | partly: real BERT/SBERT only on Kaggle |
| 5 | Embedding dimensions | test_corrected_blend (d_q), test_walkthrough_gcn_lightgbm_evaluation | partly (real encoders on Kaggle) |
| 6 | Graph construction | test_knn_*, test_follow_inductive, test_normalised_adjacency_matches_dense_formula | yes |
| 7 | GCN forward pass | test_gcn_forward_matches_dense_formula | Kaggle |
| 8 | GCN gradient updates | test_gcn_gradients_update_every_parameter | Kaggle |
| 9 | Train/val/test isolation | test_pipeline (label masking, blending), test_inductive_*, protocol audit tests | yes |
| 10 | LightGBM input dimensions | test_corrected_blend, test_walkthrough_gcn_lightgbm_evaluation | stubs here; real on Kaggle |
| 11 | Metrics | test_bot_class_metrics_by_hand, test_metric_definitions_are_consistent, test_weighted_recall_equals_accuracy | yes |
| 12 | Seed reproducibility | test_splits_disjoint_stratified_deterministic (yes), test_gcn_seed_reproducibility (Kaggle) | partly |
| 13 | Checkpoint loading | test_checkpoints_saved_only_for_requested_seeds (yes), test_checkpoint_roundtrip_reproduces_predictions (Kaggle) | partly |
| 14 | End-to-end synthetic pipeline | test_cli_*, test_suite_end_to_end_and_report (stubs), test_every_method_end_to_end_on_synthetic (Kaggle) | stubs here; real on Kaggle |

## Note on TwiBot-20 tests

Tests that still name `twibot20` exercise the library loader and the official-split code path, which are kept but not
used by the plan, notebook or paper tables. The experiment path (Cresci-2017 + Cresci-2015) is covered by the
tests marked as new or updated above.

## Mutation checks (the leakage tests can fail)

(Run before the Cresci-2015 change; the fault sites are unchanged by it.)


Each fault was injected into a throw-away copy of the current code and the full suite run:

| Injected fault | Result |
| --- | --- |
| Test labels not masked (pipeline and suite) | 15 failed |
| LightGBM early-stopped on the test set | 8 failed |
| Transductive graph used in the inductive setting | 3 failed |
| Metadata scaler fitted on all rows | 4 failed |
| Cresci-2017 manuscript literal label rule used by default | 2 failed |

## Remaining limitations

- 15 tests (all PyTorch, LightGBM and XGBoost behaviour) are unexecuted until notebook step 2 runs on Kaggle.
- Real BERT / Sentence-BERT encoders and dataset-scale run times are not tested here.
- Tests use synthetic files that copy the releases' structure; real-data checks happen in notebook steps 5-7.
