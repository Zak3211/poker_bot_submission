"""
How often does our model bust? Continued heads-up games against each dummy
bot: stacks carry over from hand to hand (starting at 500 each), the button
alternates, and a game ends after --hands hands or when either player has
no chips left. A fresh Model per game. Not part of the submission.

    .venv/bin/python scripts/bust_check.py [--hands 400] [--games 6]
        [--remaining 3000] [--set future_cap_chips=0 ...]

--genome ID plays an evolved genome (weights and genes from
evolution/hof/ID.json, else evolution/pop.json) instead of the plain model.
--remaining sets Model.hands_remaining at the start (default: --hands);
--set overrides Model.params fields, or opponents.<field> on the
opponent model (e.g. opponents.kappa_team=2).
"""

import argparse
import os
import random
import sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from arena.dummies import (CallBot, EquityBot, FoldBot, RandomBot,  # noqa: E402
                           ShoveBot, TightBot)
from card_utils import full_deck  # noqa: E402
from engine import play_hand  # noqa: E402
from model import Model  # noqa: E402

BOTS = {"fold": FoldBot, "call": CallBot, "random": RandomBot,
        "shove": ShoveBot, "tight": TightBot, "equity": EquityBot}


def game(job):
    """One continued game: (bot, seed) -> what happened."""
    name, seed, hands, stack, remaining, overrides, genome = job
    me = Model()
    if genome is not None:
        me.policy.weights = list(genome["weights"])
        for field, value in genome["genes"].items():
            setattr(me.params, field, value)
    me.hands_remaining = remaining
    me.hands_per_opponent = hands
    for field, value in overrides.items():
        if field.startswith("opponents."):  # e.g. opponents.kappa_team=2
            setattr(me.opponents, field.split(".", 1)[1], value)
        else:
            setattr(me.params, field, value)
    them = BOTS[name](seed=seed)
    stacks = [stack, stack]
    rng = random.Random(seed)
    faults = 0
    played = 0
    for h in range(hands):
        deck = full_deck()
        rng.shuffle(deck)
        result = play_hand([me, them], deck, stacks, button=h % 2,
                           teams=["us", name])
        faults += 0 in result.faults
        stacks = [s + n for s, n in zip(stacks, result.net)]
        played += 1
        if min(stacks) == 0:
            break
    return name, stacks[0] == 0, stacks[1] == 0, stacks[0], played, faults


def _load_genome(gid: int) -> dict:
    import json
    hof = os.path.join(ROOT, "evolution", "hof", f"{gid}.json")
    if os.path.exists(hof):
        with open(hof) as f:
            return json.load(f)
    with open(os.path.join(ROOT, "evolution", "pop.json")) as f:
        return next(g for g in json.load(f)["pop"] if g["id"] == gid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hands", type=int, default=400)
    parser.add_argument("--games", type=int, default=6)
    parser.add_argument("--stack", type=int, default=500)
    parser.add_argument("--bots", nargs="*", default=list(BOTS))
    parser.add_argument("--remaining", type=int, default=None)
    parser.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE")
    parser.add_argument("--genome", type=int, default=None)
    args = parser.parse_args()
    genome = _load_genome(args.genome) if args.genome is not None else None
    overrides = {k: float(v) for k, v in (kv.split("=") for kv in args.set)}
    remaining = args.remaining if args.remaining is not None else args.hands
    jobs = [(name, seed, args.hands, args.stack, remaining, overrides, genome)
            for name in args.bots for seed in range(args.games)]
    with Pool() as pool:
        results = pool.map(game, jobs, chunksize=1)
    print(f"{args.games} games of up to {args.hands} hands per bot, "
          f"stacks {args.stack}, hands_remaining {remaining}, {overrides}, "
          f"genome {args.genome if args.genome is not None else 'none (plain model)'}")
    print(f"{'bot':8} {'we bust':>8} {'they bust':>10} {'mean final':>11} "
          f"{'min final':>10} {'mean hands':>11} {'faults':>7}")
    total_busts = 0
    for name in args.bots:
        rows = [r for r in results if r[0] == name]
        busts = sum(r[1] for r in rows)
        total_busts += busts
        finals = [r[3] for r in rows]
        print(f"{name:8} {busts:>5}/{len(rows):<2} {sum(r[2] for r in rows):>7}/{len(rows):<2}"
              f" {sum(finals) / len(rows):>11.0f} {min(finals):>10}"
              f" {sum(r[4] for r in rows) / len(rows):>11.0f}"
              f" {sum(r[5] for r in rows):>7}")
    print(f"total: we bust in {total_busts} of {len(results)} games")


if __name__ == "__main__":
    main()
