"""hand_history.py: the replayed hand matches the real one."""

import random

import pytest

from card_utils import full_deck
from cards import card_id
from engine import play_hand
from hand_history import HandHistory
from player_action import PlayerAction

from arena.dummies import CallBot, FoldBot, RandomBot, ShoveBot


class Checker(RandomBot):
    """A random bot that replays the hand and compares it each turn."""

    def __init__(self, seed=None):
        super().__init__(seed)
        self.history = HandHistory()
        self.problems = []
        self.start_stacks = None

    def seed(self, hand, player_id):
        super().seed(hand, player_id)
        self.history.start([0, 1], player_id)  # cards do not matter here

    def run(self, actions, game_state):
        self.history.observe(actions, game_state)
        self._compare(game_state)
        return super().run(actions, game_state)

    def end_game(self, actions, game_state):
        self.history.finish(actions, game_state)
        self._compare(game_state)

    def _compare(self, state):
        h = self.history
        if h._pot != state.pot:
            self.problems.append(("pot", h._pot, state.pot))
        if list(h._stacks) != list(state.stacks):
            self.problems.append(("stacks", h._stacks, state.stacks))


@pytest.mark.parametrize("opponent", [RandomBot, CallBot, ShoveBot, FoldBot])
@pytest.mark.parametrize("stacks", [[500, 500], [130, 870], [905, 15]])
def test_replay_matches_state(opponent, stacks):
    for seed in range(15):
        deck = full_deck()
        random.Random(seed).shuffle(deck)
        me = Checker(seed=seed)
        result = play_hand([me, opponent(seed=seed + 1)], deck, list(stacks),
                           button=seed % 2)
        assert result.faults == {}
        assert me.problems == []
        assert me.history.start_stacks == stacks
        assert me.history.board == [card_id(c) for c in result.board]


def test_records_situations():
    h = HandHistory()
    h.start([0, 1], 1)

    class State:
        num_players, teams, button = 2, ("a", "b"), 0
        small_blind, big_blind = 10, 20
        stacks = [440, 480]  # after the moves below
        pot = 80

    # Button (small blind) raises to 60, big blind re-raises to 200.
    h.observe([PlayerAction.raise_to(0, 60)], State)
    assert h.start_stacks == [500, 500]
    h._apply(PlayerAction.raise_to(1, 200))
    h._apply(PlayerAction.call(0))
    r0, r1, r2 = h.records
    assert (r0.kind, r0.facing, r0.pot_before, r0.raise_by) == ("raise", 10, 30, 40)
    assert r0.size_frac == pytest.approx(40 / 40)
    assert (r1.kind, r1.facing, r1.aggressor, r1.raise_by) == ("raise", 40, False, 140)
    assert (r2.kind, r2.amount, r2.aggressor) == ("call", 140, True)  # raised before
    assert h._pot == 400 and not r2.all_in
