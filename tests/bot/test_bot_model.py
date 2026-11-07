"""The model plays legally heads-up, at any stack depth."""

import pickle
import random

import pytest

from card_utils import full_deck
from engine import play_hand
from decision import push_fold
from model import Model

from arena.dummies import (CallBot, EquityBot, FoldBot, RandomBot, ShoveBot,
                           TightBot)


@pytest.mark.parametrize("opponent", [FoldBot, CallBot, ShoveBot, RandomBot,
                                      TightBot, EquityBot])
@pytest.mark.parametrize("stacks", [[500, 500], [60, 940], [1500, 90]])
def test_plays_legally(opponent, stacks):
    me = Model()
    me.debug = True
    for seed in range(8):
        deck = full_deck()
        random.Random(seed).shuffle(deck)
        seats = [me, opponent(seed=seed)]
        result = play_hand(seats, deck, stacks, button=seed % 2)
        assert 0 not in result.faults


class PushFolder(Model):
    def run(self, actions, game_state):
        self.history.observe(actions, game_state)
        return push_fold(game_state, self.player_id, self.history.hand,
                         self.rng)


def test_push_fold_never_limps():
    me = PushFolder()
    for seed in range(40):
        deck = full_deck()
        random.Random(seed).shuffle(deck)
        result = play_hand([me, CallBot()], deck, [500, 500], button=0)
        first = next(a for a in result.actions if a.player_id == 0)
        assert first.kind in ("fold", "raise")


def test_pickles_after_play():
    me = Model()
    deck = full_deck()
    random.Random(0).shuffle(deck)
    play_hand([me, CallBot()], deck, [500, 500])
    assert pickle.loads(pickle.dumps(me)).hands_played == 1
