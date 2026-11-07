"""opponents.py: the generative opponent model, range queries, learning."""

import pickle
import random

import pytest

from card_utils import full_deck
from cards import COMBOS, NUM_COMBOS
from engine import play_hand
from hand_history import HandHistory
from model import Model
from opponents import (ALLIN, CHECK, FOLD, OUTCOMES, ActionModel,
                       OpponentModel, Situation, base_probs, legal, outcome_of)
from player_action import PlayerAction
from strength import NUM_BUCKETS, StrengthCache

from arena.dummies import CallBot, EquityBot, ShoveBot


def all_situations():
    for street in ("preflop", "flop", "turn", "river"):
        for facing in ("none", "small", "medium", "large"):
            for aggressor in (False, True):
                yield Situation(street, facing, aggressor)


def test_base_probs_are_distributions_and_monotone():
    for sit in all_situations():
        rows = [base_probs(sit, b) for b in range(NUM_BUCKETS)]
        for p in rows:
            assert len(p) == len(OUTCOMES)
            assert sum(p) == pytest.approx(1)
            assert min(p) >= 0
        raises = [sum(p[3:]) for p in rows]
        assert raises == sorted(raises)  # stronger raises more
        if sit.facing != "none":
            folds = [p[FOLD] for p in rows]
            assert folds == sorted(folds, reverse=True)


def test_situation_levels():
    assert Situation.of("flop", 0, 100, False).facing == "none"
    assert Situation.of("flop", 30, 130, False).facing == "small"  # ~0.3 pot bet
    assert Situation.of("flop", 50, 150, False).facing == "medium"  # half pot
    assert Situation.of("flop", 100, 200, False).facing == "large"  # pot


def play(me, bot, hands, name):
    for h in range(hands):
        deck = full_deck()
        random.Random(h).shuffle(deck)
        play_hand([me, bot(seed=h)], deck, [500, 500], button=h % 2,
                  teams=["us", name])


class Caller(Model):
    """Learns as Model, but checks or calls everything: many showdowns."""

    def run(self, actions, game_state):
        self.history.observe(actions, game_state)
        return PlayerAction.call(self.player_id)


def test_learns_a_shover():
    me = Caller()
    me.debug = True
    play(me, ShoveBot, 30, "shove")
    om = me.opponents
    sit = Situation("preflop", "medium", False)  # the button owes 10 into 30
    assert om.trust("shove", sit) > 0.3
    for b in range(NUM_BUCKETS):
        assert om.action_probs("shove", sit, b)[ALLIN] > 0.6
    # An unknown team still gets the population's (now shove-heavy) view,
    # with no trust.
    assert om.trust("someone else", sit) == 0


def test_learns_a_calling_station_never_folds():
    me = Caller()
    me.debug = True
    play(me, CallBot, 40, "call")
    sit = Situation("flop", "none", False)
    for b in range(NUM_BUCKETS):
        p = me.opponents.action_probs("call", sit, b)
        assert p[CHECK] > 0.9


def test_hand_distribution_is_a_distribution_without_dead_cards():
    me = Caller()
    me.debug = True
    play(me, EquityBot, 20, "equity")
    h = me.history
    opp = h.opponent()
    weights = me.opponents.query_hand_distribution(
        "equity", h, opp, StrengthCache())
    assert len(weights) == NUM_COMBOS
    assert sum(weights) == pytest.approx(1)
    dead = set(h.hand + h.board)
    for w, (a, b) in zip(weights, COMBOS):
        if a in dead or b in dead:
            assert w == 0


def test_respond_is_consistent():
    om = OpponentModel()
    sit = Situation("flop", "large", False)
    weights = [1.0] * NUM_COMBOS
    buckets = [i % NUM_BUCKETS for i in range(NUM_COMBOS)]
    r = om.respond("x", sit, weights, buckets, can_raise=True)
    assert r.fold + r.call + r.raise_ == pytest.approx(1)
    assert 0 < r.fold < 1
    # Continuing weight is the range times the chance of not folding:
    # stronger buckets keep more.
    by_bucket = [r.continuing[i] for i in range(NUM_BUCKETS)]
    assert by_bucket == sorted(by_bucket)
    r = om.respond("x", sit, weights, buckets, can_raise=False)
    assert r.raise_ == pytest.approx(0, abs=1e-9)


def test_soft_counts_add_up():
    m = ActionModel()
    sit = Situation("river", "none", True)
    m.add(sit, [0.2, 0.3, 0.5, 0, 0], CHECK)
    assert m.n(sit) == pytest.approx(1)
    assert m.marginal(sit)[CHECK] == pytest.approx(1)
    assert m.cell(sit, 3) is None


def test_model_with_opponents_pickles():
    me = Caller()
    play(me, EquityBot, 10, "equity")
    clone = pickle.loads(pickle.dumps(me))
    sit = Situation("flop", "none", False)
    assert clone.opponents.action_probs("equity", sit, 4) == \
        me.opponents.action_probs("equity", sit, 4)
    assert len(pickle.dumps(me)) < 200_000
