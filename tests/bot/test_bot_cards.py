"""cards.py: ids, classes and the fast evaluator; the generated tables."""

from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from card_utils import Card, evaluate, full_deck
from cards import (CLASS_OF, COMBOS, DECK, NUM_COMBOS, card_id, category_of,
                   class_name, combo_id, encode_rank, hand_class, rank7)
from tables import CALL, EQ169, MAX_BB, PUSH


def ids(text: str) -> list[int]:
    return [card_id(Card.parse(t)) for t in text.split()]


def test_card_ids_follow_full_deck():
    assert [card_id(c) for c in full_deck()] == list(range(52))


def test_classes_partition_combos():
    counts = Counter(CLASS_OF)
    kinds = Counter(
        ("pair" if len(class_name(c)) == 2 else class_name(c)[2], n)
        for c, n in counts.items())
    assert kinds == {("pair", 6): 13, ("s", 4): 78, ("o", 12): 78}
    assert class_name(hand_class(*ids("AS KS"))) == "AKs"
    assert class_name(hand_class(*ids("KD AS"))) == "AKo"
    for i, (a, b) in enumerate(COMBOS):
        assert combo_id(b, a) == i
    assert len(COMBOS) == NUM_COMBOS


@settings(max_examples=3000, deadline=None)
@given(st.lists(st.integers(0, 51), min_size=5, max_size=7, unique=True))
def test_rank7_orders_as_evaluate(cards):
    assert rank7(cards) == encode_rank(evaluate([DECK[i] for i in cards]))


@settings(max_examples=500, deadline=None)
@given(st.lists(st.integers(0, 51), min_size=7, max_size=7, unique=True),
       st.lists(st.integers(0, 51), min_size=7, max_size=7, unique=True))
def test_rank7_compares_as_evaluate(a, b):
    ea, eb = (evaluate([DECK[i] for i in x]) for x in (a, b))
    ra, rb = rank7(a), rank7(b)
    assert (ra > rb) == (ea > eb) and (ra == rb) == (ea == eb)


def test_rank7_examples():
    assert category_of(rank7(ids("AS 2S 3S 4S 5S"))) == 8  # steel wheel
    assert rank7(ids("AS 2D 3C 4H 5S 9D KC")) == encode_rank((4, 5))
    # Three pairs: the third pair's rank is the kicker if it is highest.
    assert rank7(ids("2S 2D 3C 3H KS KD QC")) == encode_rank((2, 13, 3, 12))
    # Two trips make a full house.
    assert rank7(ids("2S 2D 2C 9H 9S 9D AC")) == encode_rank((6, 9, 2))


def test_equity_table():
    aa, kk, ak_s, deuces = 168, 154, 12 * 13 + 11, 0
    assert abs(EQ169[aa][kk] - 0.82) < 0.02
    assert abs(EQ169[ak_s][deuces] - 0.50) < 0.03
    for i in range(169):
        assert EQ169[i][i] == 0.5
        for j in range(169):
            assert abs(EQ169[i][j] + EQ169[j][i] - 1) < 1e-3


def test_push_fold_tables():
    aa, seven_deuce = 168, 0 * 13 + 5  # 72o is at (low=0, high=5)
    assert class_name(seven_deuce) == "72o"
    for e in range(1, MAX_BB + 1):
        assert PUSH[e][aa] > 0.99 and CALL[e][aa] > 0.99
    # Deeper stacks push and call less.
    weight = [CLASS_OF.count(c) for c in range(169)]

    def share(row):
        return sum(w * x for w, x in zip(weight, row)) / NUM_COMBOS

    assert share(PUSH[5]) > share(PUSH[15]) > share(PUSH[25])
    assert share(CALL[5]) > share(CALL[15]) > share(CALL[25])
