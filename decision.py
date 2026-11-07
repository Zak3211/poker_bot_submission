"""
Choosing a move. Step 1: heads-up push/fold equilibrium preflop (tables.py)
and a simple made-hand rule after the flop.
"""

import random

from cards import category_of, hand_class, rank7
from player_action import PlayerAction
from tables import CALL, MAX_BB, PUSH


def effective_bb(state, me: int) -> int:
    """Effective stack in big blinds, chips already bet included, as the
    push/fold tables count it: 1..MAX_BB."""
    other = 1 - me if state.num_players == 2 else max(
        (p for p in state.players_in_hand if p != me),
        key=lambda p: state.stacks[p] + state.bets[p])
    chips = min(state.stacks[me] + state.bets[me],
                state.stacks[other] + state.bets[other])
    return max(1, min(MAX_BB, round(chips / state.big_blind)))


def preflop(state, me: int, hand: list[int], rng: random.Random) -> PlayerAction:
    cls = hand_class(*hand)
    e = effective_bb(state, me)
    shove = PlayerAction.raise_to(me, state.max_raise_to) if state.can_raise else None
    opener = state.bets[me] < state.big_blind and state.current_bet == state.big_blind
    if opener:  # small blind, first in: shove or fold
        if shove and rng.random() < PUSH[e][cls]:
            return shove
        return PlayerAction.fold(me)
    # Facing a shove, a raise, or (big blind) a limp: our calling range.
    strong = rng.random() < CALL[e][cls]
    if state.call_amount == 0:
        return shove if strong and shove else PlayerAction.call(me)
    if not strong:
        return PlayerAction.fold(me)
    return shove or PlayerAction.call(me)


def postflop(state, me: int, hand: list[int], board: list[int]) -> PlayerAction:
    category = category_of(rank7(hand + board))
    board_category = category_of(rank7(board)) if len(board) >= 5 else 0
    if category >= 2 and category > board_category and state.can_raise:
        low, high = state.raise_range
        return PlayerAction.raise_to(me, min(high, max(low, state.pot)))
    if state.call_amount == 0:
        return PlayerAction.call(me)
    if category >= 1 and state.pot_odds < 0.3:
        return PlayerAction.call(me)
    return PlayerAction.fold(me)
