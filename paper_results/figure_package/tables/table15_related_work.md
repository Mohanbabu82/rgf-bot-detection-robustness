### TABLE 15. Positioning against related work

| Study | Modalities | Approach | Reported finding | Relation to this work |
|---|---|---|---|---|
| BotRGCN [Feng et al., 2021] | metadata, text, follow graph | relational GCN | TwiBot-20 acc 0.8462, F1 0.8707 (reported) | R-GCN baseline re-implemented here |
| RGT [Feng et al., 2022] | metadata, text, heterogeneous graph | relation-aware graph transformer | outperforms prior methods on TwiBot-20 | not re-run |
| BotMoE [Liu et al., 2023] | metadata, text, graph | community-aware modality experts | most robust to feature manipulation in its study | closest architecture; not re-run |
| BotBuster [Ng & Carley, 2023] | per-platform account fields | mixture of experts | handles incomplete account information | missing-input motivation |
| BotCHF [Wang et al., 2026] | multi-modal | camouflage-aware fusion | fusion weights differ across datasets | per-account weighting is not new |
| LMBot [Cai et al., 2024] | text, graph (distilled) | graph-to-LM distillation | graph-less deployment at state-of-the-art accuracy | alternative to graph availability |
| Hays et al., 2023 | simple features | shallow decision trees | near state-of-the-art; datasets do not generalise | consistent with our strong tree baselines |
| Echeverría et al., 2018 | account features | leave-one-botnet-out | >97% accuracy does not transfer to unseen bot classes | consistent with our cross-dataset result |
| Mukherjee et al., 2026 | graph | optimal-transport edge attacks | attacks bypass five defences | edge injection is a real threat |
| This work | metadata, text, graph | availability-masked gated experts | see Tables 4-14 | same protocol for all baselines |

*Findings are summarised from the verified reference library (VERIFIED_REFERENCES.csv); numbers from other papers use different datasets, splits and protocols and are not comparable with Tables 4-14, so no numeric ranking is given.*
