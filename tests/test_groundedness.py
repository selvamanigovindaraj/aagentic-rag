from app.services.groundedness import is_grounded


def test_short_entity_is_grounded_by_substring_when_it_has_no_significant_words():
    assert is_grounded("3M", "3M reported record earnings.")
    assert not is_grounded("3M", "Acme reported record earnings.")


def test_grounded_candidate_shares_a_significant_word_with_source():
    assert is_grounded("recursive summaries", "RAPTOR builds summaries.")


def test_ungrounded_candidate_shares_no_significant_word_with_source():
    assert not is_grounded(
        "click here for a free prize", "RAPTOR builds recursive summary trees."
    )
