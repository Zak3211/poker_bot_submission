import random

from card_utils import Card
from cards import card_id
from decision import postflop, preflop
from game_state import GameStateView
from player_action import DEALER, PlayerAction


class Model:
    """
    Heads-up bot. Step 1: push/fold equilibrium preflop (decision.py,
    tables.py), a simple made-hand rule after the flop.

    This object is what gets pickled, so everything here is plain data.
    """

    def __init__(self):
        self.hand = []
        self.player_id = None
        self.hand_ids: list[int] = []
        self.board: list[int] = []
        self.rng = random.Random(0)
        self.hands_played = 0
        self.debug = False  # True: errors in run() raise instead of check/fold

    def seed(self, hand: list[Card], player_id: int):
        self.hand = hand
        self.player_id = player_id
        self.hand_ids = [card_id(c) for c in hand]
        self.board = []

    def run(
        self, actions: list[PlayerAction], game_state: GameStateView
    ) -> PlayerAction:
        try:
            for action in actions:
                if action.player_id == DEALER:
                    self.board.append(card_id(action.card))
            if game_state.street == "preflop":
                return preflop(game_state, self.player_id, self.hand_ids,
                               self.rng)
            return postflop(game_state, self.player_id, self.hand_ids,
                            self.board)
        except Exception:
            if self.debug:
                raise
            # Never worse than a fault: check if free, else fold.
            if game_state.call_amount == 0:
                return PlayerAction.call(self.player_id)
            return PlayerAction.fold(self.player_id)

    def end_game(
        self, actions: list[PlayerAction], game_state: GameStateView
    ):
        self.hands_played += 1
