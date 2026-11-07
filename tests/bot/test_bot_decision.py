"""equity.py and decision.py: equity vectors and expected-chips options."""

import random

import pytest

from card_utils import Card, evaluate, full_deck
from cards import COMBOS, DECK, NUM_COMBOS, card_id, combo_id
from decision import Params, evaluate_options
from engine import play_hand
from equity import equity_vector, range_equity
from model import Model
from opponents import OpponentModel
from tables import EQ169

from arena.dummies import CallBot


def ids(text):
    return [card_id(Card.parse(t)) for t in text.split()]


def brute(hand, board, combo):
    """Our showdown share against combo over every runout."""
    from itertools import combinations
    dead = set(hand) | set(board) | set(combo)
    rest = [c for c in range(52) if c not in dead]
    won = n = 0
    for runout in combinations(rest, 5 - len(board)):
        full = [DECK[i] for i in board + list(runout)]
        ours = evaluate([DECK[i] for i in hand] + full)
        theirs = evaluate([DECK[i] for i in combo] + full)
        won += 1 if ours > theirs else 0.5 if ours == theirs else 0
        n += 1
    return won / n


@pytest.mark.parametrize("board", ["2C 7D 9H TS 3S", "2C 7D 9H TS"])
def test_equity_vector_exact_on_turn_and_river(board):
    hand, board = ids("AS TD"), ids(board)
    eq = equity_vector(hand, board, random.Random(0))
    rng = random.Random(1)
    live = [i for i, (a, b) in enumerate(COMBOS)
            if not {a, b} & set(hand + board)]
    for i in rng.sample(live, 15):
        assert eq[i] == pytest.approx(brute(hand, board, list(COMBOS[i])))
    assert eq[combo_id(*ids("AS 2D"))] == 0.0  # holds our ace: impossible


def test_equity_vector_preflop_uses_the_table():
    hand = ids("AS AH")
    eq = equity_vector(hand, [], random.Random(0))
    kk = combo_id(*ids("KS KH"))
    assert eq[kk] == EQ169[168][154]
    possible = [0.0 if {a, b} & set(hand) else 1.0 for a, b in COMBOS]
    assert range_equity(eq, possible) == pytest.approx(0.85, abs=0.01)


class Inspector(Model):
    """Records the options considered at each decision."""

    def __init__(self):
        super().__init__()
        self.debug = True
        self.seen = []

    def run(self, actions, game_state):
        self.history.observe(actions, game_state)
        options = evaluate_options(game_state, self.player_id, self.history,
                                   self.opponents, self.strengths,
                                   self.params, self.rng)
        self.seen.append((game_state.call_amount, options))
        return max(options, key=lambda o: o.ev).action


def test_options_are_legal_and_sane():
    me = Inspector()
    for h in range(15):
        deck = full_deck()
        random.Random(h).shuffle(deck)
        result = play_hand([me, CallBot()], deck, [500, 500], button=h % 2)
        assert result.faults == {}
    for owe, options in me.seen:
        kinds = [o.action.kind for o in options]
        assert ("fold" in kinds) == (owe > 0)  # never fold a free check
        for o in options:
            if o.action.kind == "fold":
                assert o.ev == 0.0
            assert 0.0 <= o.fold_p <= 1.0


def test_expects_a_calling_station_to_call():
    """After playing a calling station, the raises we consider expect it
    to fold rarely, even in situations it was seldom seen in."""
    me = Inspector()
    me.opponents = OpponentModel()
    # Teach the model a calling station by playing one.
    for h in range(80):
        deck = full_deck()
        random.Random(h).shuffle(deck)
        play_hand([me, CallBot()], deck, [500, 500], button=h % 2,
                  teams=["us", "station"])
    raises = []
    for owe, options in me.seen[-60:]:
        for o in options:
            if o.action.kind == "raise":
                raises.append(o.fold_p)
    assert raises and sum(raises) / len(raises) < 0.12
