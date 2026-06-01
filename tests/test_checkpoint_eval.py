from cplab.eval.checkpoint import checkpoint_deltas


def test_checkpoint_deltas_use_metric_direction() -> None:
    deltas = checkpoint_deltas(
        {
            "domain_benchmark": {
                "surface": 10.0,
                "recall_exact_match": 0.25,
                "application_token_f1": 0.4,
            },
            "general_retention": {"general_perplexity": 20.0},
        },
        {
            "domain_benchmark": {
                "surface": 7.5,
                "recall_exact_match": 0.5,
                "application_token_f1": 0.3,
            },
            "general_retention": {"general_perplexity": 21.0},
        },
    )

    assert deltas["domain_surface_perplexity_delta"] == -2.5
    assert deltas["domain_surface_gain"] == 2.5
    assert deltas["domain_recall_exact_match_delta"] == 0.25
    assert deltas["domain_application_token_f1_delta"] == -0.10000000000000003
    assert deltas["general_perplexity_delta"] == 1.0
    assert deltas["general_retention_delta"] == -1.0
