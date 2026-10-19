"""Testes do mapeamento de categorias para níveis de urgência."""

from triage.labels import LEVELS, URGENCY_MAP


def test_all_25_categories_are_mapped():
    assert len(URGENCY_MAP) == 25


def test_only_the_three_levels_are_used():
    assert set(URGENCY_MAP.values()) == set(LEVELS)
