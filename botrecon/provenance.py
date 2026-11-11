"""Provenance labels and externally sourced numbers.

Numbers in this file are NOT produced by this code. They are copied, with
their source, so that report tables can show them next to reproduced numbers
without ever being mixed into a reproduced row.
"""

MANUSCRIPT_REPORTED = "MANUSCRIPT-REPORTED"
REPRODUCED_LITERAL = "REPRODUCED-LITERAL"
REPRODUCED_CORRECTED = "REPRODUCED-CORRECTED"
DIAGNOSTIC_INVALID = "DIAGNOSTIC-INVALID"
LITERATURE = "LITERATURE"
SMOKE_TEST = "SMOKE-TEST-NOT-A-RESULT"
THIS_WORK = "THIS-WORK"          # new experiments of the revised paper (proposed method, baselines, ablations)

ALL_LABELS = (
    MANUSCRIPT_REPORTED,
    REPRODUCED_LITERAL,
    REPRODUCED_CORRECTED,
    DIAGNOSTIC_INVALID,
    LITERATURE,
    SMOKE_TEST,
    THIS_WORK,
)

# Values as printed in the manuscript (percent). The manuscript does not state
# its averaging convention reliably; recall == accuracy on TwiBot-20 suggests
# weighted averaging. Implementation lost; not independently verified.
MANUSCRIPT_RESULTS = {
    "cresci2017": {
        "accuracy": 98.57, "precision": 98.59, "recall": 98.58, "f1": 98.58,
        "averaging": "unstated (apparently weighted)",
        "source": "Manuscript, Table 2",
    },
    "twibot20": {
        "accuracy": 92.28, "precision": 92.82, "recall": 92.28, "f1": 92.24,
        "averaging": "unstated (apparently weighted)",
        "source": "Manuscript, Table 3",
    },
}

MANUSCRIPT_ABLATION_TWIBOT20_F1 = {
    "metadata_only": 86.4,
    "metadata_bert": 89.1,
    "metadata_gcn": 90.5,
    "metadata_bert_gcn": 92.2,
}

# Published TwiBot-20 results (official split), copied from the sources below.
# Only values that were opened in the source paper are included.
LITERATURE_TWIBOT20 = [
    {
        "model": "BotRGCN", "accuracy": 84.62, "f1": 87.07, "mcc": 70.21,
        "source": "Feng et al., BotRGCN, ASONAM 2021, Table III",
        "url": "https://arxiv.org/pdf/2106.13092",
    },
    {
        "model": "SATAR", "accuracy": 84.12, "f1": 86.42, "mcc": 68.63,
        "source": "Reported in Feng et al., BotRGCN, ASONAM 2021, Table III",
        "url": "https://arxiv.org/pdf/2106.13092",
    },
    {
        "model": "Alhosseini et al. (re-implemented)", "accuracy": 68.13,
        "f1": 73.18, "mcc": 35.43,
        "source": "Reported in Feng et al., BotRGCN, ASONAM 2021, Table III",
        "url": "https://arxiv.org/pdf/2106.13092",
    },
    {
        "model": "RGT", "accuracy": 86.64, "f1": 88.21, "mcc": None,
        "source": "Feng et al., RGT, AAAI 2022",
        "url": "https://arxiv.org/pdf/2109.02927",
    },
]
