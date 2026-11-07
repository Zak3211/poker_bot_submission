"""
Steady-state neuroevolution of the policy network's output layer
(policy_net.py) and a few decision Params. Not part of the submission.

Fitness is paired against genome 0 (the engine as it is, output layer at
zero), so card luck mostly cancels:

- paired jobs: --decks decks against one opponent (a hall-of-fame genome
  with chance --hof-share, else a dummy), each deck played twice with seats swapped (cards
  and position), stacks reset every hand. Genome 0 plays the same decks
  against the same opponent; the score is the difference in bb/100.
  Genome 0's results are deterministic, so they are cached per (opponent,
  deck block); blocks come from a fixed pool so the cache is reused.
- bust jobs (--bust-share of them): a continued game of up to --hands hands
  (stacks carry over), for the genome and genome 0 on the same seed; the
  score is the difference in busts.

fitness = mean paired bb/100 - --bust-weight * (bust rate - genome 0's).

A router keeps every worker busy with no generation barrier: genomes short
of --min-jobs paired jobs go first, then re-evaluations of the top, then
children of tournament winners replacing the worst.

Selection and culling rank by a conservative score, fitness minus one
standard error, so a well-tested genome is not culled for a newcomer whose
mean is high only by luck; the best by that score is never culled.
Stopping the run (SIGTERM or Ctrl-C) saves a snapshot first.

    .venv/bin/python scripts/evolve.py --minutes 30 [--resume]
    .venv/bin/python scripts/evolve.py --h2h [--top 4 --games 20]
    .venv/bin/python scripts/evolve.py --export best [--force]

State: evolution/pop.json (rewritten atomically), evolution/hof/<id>.json.
--export only writes policy_weights.py when the genome beats genome 0 by
more than 1.96 standard errors (or with --force).
"""

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from policy_net import PolicyNet  # noqa: E402

OUT = os.path.join(ROOT, "evolution")
DUMMIES = ["fold", "call", "random", "shove", "tight", "equity"]
# How often each dummy is drawn for paired jobs: the realistic, informative
# ones (equity-threshold, calling station, erratic) more, fold (played the
# same by every genome) least. Bust jobs draw uniformly.
DUMMY_WEIGHTS = {"equity": 3, "call": 2, "random": 2, "tight": 1,
                 "shove": 1, "fold": 0.5}
GENE_RANGES = {
    "realisation": (0.6, 1.0),
    "survival_scale_bb": (3.0, 30.0),
    "future_cap_chips": (0.0, 15000.0),
}
BB = 20


@dataclass
class Genome:
    id: int
    weights: list[float]
    genes: dict[str, float]
    parents: list[int] = field(default_factory=list)
    paired: dict[str, list[float]] = field(default_factory=dict)  # bb/100 diffs per opponent
    bust_diff: list[float] = field(default_factory=list)  # busts - genome 0's
    in_flight: int = 0  # running jobs; not saved

    def diffs(self) -> list[float]:
        return [x for v in self.paired.values() for x in v]

    def jobs(self) -> int:
        return len(self.diffs())

    def fitness(self, bust_weight: float) -> float:
        d = self.diffs()
        if not d:
            return -math.inf
        bust = sum(self.bust_diff) / len(self.bust_diff) if self.bust_diff else 0.0
        return sum(d) / len(d) - bust_weight * bust

    def stderr(self) -> float:
        d = self.diffs()
        if len(d) < 2:
            return math.inf
        mean = sum(d) / len(d)
        var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
        return math.sqrt(var / len(d))

    def to_json(self) -> dict:
        return {"id": self.id, "weights": self.weights, "genes": self.genes,
                "parents": self.parents, "paired": self.paired,
                "bust_diff": self.bust_diff}

    @classmethod
    def from_json(cls, d: dict) -> "Genome":
        return cls(d["id"], d["weights"], d["genes"], d["parents"],
                   d.get("paired", {}), d.get("bust_diff", []))


# -- worker side ----------------------------------------------------------------

def _init_worker():
    # Workers fork after the run's stop handler is set: give them the
    # default handlers back so stopping the run kills them.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    import strength
    strength.FLOP_RUNOUTS = 15  # cheaper strength estimates while training


def _build(weights, genes):
    from model import Model
    m = Model()
    m.policy.weights = list(weights)
    for name, value in genes.items():
        setattr(m.params, name, value)
    m.params.flop_runouts = 20
    m.hands_remaining = 3000
    m.hands_per_opponent = 200
    return m


def _opponent(opp: dict, seed: int):
    from arena.dummies import (CallBot, EquityBot, FoldBot, RandomBot,
                               ShoveBot, TightBot)
    bots = {"fold": FoldBot, "call": CallBot, "random": RandomBot,
            "shove": ShoveBot, "tight": TightBot, "equity": EquityBot}
    if opp["kind"] == "dummy":
        return bots[opp["name"]](seed=seed), opp["name"]
    return _build(opp["weights"], opp["genes"]), f"g{opp['id']}"


def _duplicate(weights, genes, opp: dict, seed: int, decks: int,
               stack: int) -> float:
    """bb/100 over decks, each played twice with seats (cards and
    position) swapped, stacks reset every hand."""
    from card_utils import full_deck
    from engine import play_hand
    me = _build(weights, genes)
    them, name = _opponent(opp, seed)
    rng = random.Random(seed)
    net = 0
    for _ in range(decks):
        deck = full_deck()
        rng.shuffle(deck)
        a = play_hand([me, them], deck, [stack, stack], button=0,
                      teams=["us", name])
        b = play_hand([them, me], deck, [stack, stack], button=0,
                      teams=[name, "us"])
        net += a.net[0] + b.net[1]
    return net / BB / (2 * decks) * 100


def _continued(weights, genes, opp: dict, seed: int, hands: int,
               stack: int) -> bool:
    """A continued game (stacks carry over): did we bust?"""
    from card_utils import full_deck
    from engine import play_hand
    me = _build(weights, genes)
    them, name = _opponent(opp, seed)
    stacks = [stack, stack]
    rng = random.Random(seed)
    for h in range(hands):
        deck = full_deck()
        rng.shuffle(deck)
        result = play_hand([me, them], deck, stacks, button=h % 2,
                           teams=["us", name])
        stacks = [s + n for s, n in zip(stacks, result.net)]
        if min(stacks) == 0:
            break
    return stacks[0] == 0


def play_job(job: dict) -> dict:
    """One job for a genome, and for genome 0 on the same cards unless its
    result is cached. Runs in a worker."""
    run = _duplicate if job["type"] == "paired" else _continued
    size = job["decks"] if job["type"] == "paired" else job["hands"]
    out = {"job": job["key"], "id": job["id"], "type": job["type"],
           "opp": job["opp_label"]}
    out["value"] = run(job["weights"], job["genes"], job["opp"], job["seed"],
                       size, 500)
    if job.get("base") is not None:
        b = job["base"]
        out["base"] = run(b["weights"], b["genes"], job["opp"], job["seed"],
                          size, 500)
    return out


# -- evolution ------------------------------------------------------------------

class Evolution:
    def __init__(self, args):
        self.args = args
        self.rng = random.Random(args.seed)
        self.pop: dict[int, Genome] = {}
        self.hof: list[Genome] = []
        self.next_id = 0
        self.base_cache: dict[str, float] = {}  # job key -> genome 0's result
        self.pending: dict[str, list[dict]] = {}  # job key -> results waiting on genome 0
        self.jobs_done = self.hands = 0

    # state ---------------------------------------------------------------

    def seed_population(self):
        from decision import Params
        base = Params()
        genes = {k: float(getattr(base, k)) for k in GENE_RANGES}
        net = PolicyNet.zero_output(random.Random(self.args.seed))
        self.pop[0] = Genome(self._new_id(), list(net.weights), dict(genes))
        while len(self.pop) < self.args.pop:
            child = Genome(self._new_id(), list(net.weights), dict(genes), [0])
            self._mutate(child)
            self.pop[child.id] = child

    def save(self):
        os.makedirs(OUT, exist_ok=True)
        state = {"next_id": self.next_id, "base_cache": self.base_cache,
                 "pop": [g.to_json() for g in self.pop.values()],
                 "hof": [g.id for g in self.hof]}
        tmp = os.path.join(OUT, "pop.json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, os.path.join(OUT, "pop.json"))

    def load(self):
        with open(os.path.join(OUT, "pop.json")) as f:
            state = json.load(f)
        self.next_id = state["next_id"]
        self.base_cache = state.get("base_cache", {})
        self.pop = {d["id"]: Genome.from_json(d) for d in state["pop"]}
        for gid in state["hof"]:
            with open(os.path.join(OUT, "hof", f"{gid}.json")) as f:
                self.hof.append(Genome.from_json(json.load(f)))

    def to_hof(self, g: Genome):
        os.makedirs(os.path.join(OUT, "hof"), exist_ok=True)
        with open(os.path.join(OUT, "hof", f"{g.id}.json"), "w") as f:
            json.dump(g.to_json(), f)
        self.hof = [h for h in self.hof if h.id != g.id] + [Genome.from_json(g.to_json())]
        self.hof = self.hof[-self.args.hof_size:]  # the most recent entries

    # genetics -----------------------------------------------------------

    def _new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1

    def _mutate(self, g: Genome):
        """Gaussian noise on the output layer (the hidden layer stays
        frozen) and log-normal steps on the genes."""
        r = self.args
        net = PolicyNet()
        start = 0 if not r.freeze_hidden else net.hidden * net.inputs + net.hidden
        g.weights = [w + self.rng.gauss(0, r.sigma)
                     if i >= start and self.rng.random() < r.mut_rate else w
                     for i, w in enumerate(g.weights)]
        for name, (low, high) in GENE_RANGES.items():
            if self.rng.random() < 0.3:
                value = g.genes[name]
                if name == "future_cap_chips":
                    value += self.rng.gauss(0, 1000)
                else:
                    value *= math.exp(self.rng.gauss(0, 0.15))
                g.genes[name] = min(high, max(low, value))

    def fit(self, g: Genome) -> float:
        return g.fitness(self.args.bust_weight)

    def score(self, g: Genome) -> float:
        """Conservative: fitness minus one standard error."""
        return self.fit(g) - g.stderr()

    def evaluated(self) -> list[Genome]:
        return [g for g in self.pop.values()
                if g.id != 0 and g.jobs() >= self.args.min_jobs]

    def _tournament(self) -> Genome:
        pool = self.evaluated()
        return max(self.rng.sample(pool, min(3, len(pool))), key=self.score)

    def breed(self) -> Genome | None:
        done = self.evaluated()
        if len(done) < 3:
            return None
        if len(self.pop) >= self.args.pop:
            # The worst by the conservative score goes, never the best;
            # results still running for it are dropped (genome 0's kept).
            elite = max(done, key=self.score)
            removable = [g for g in done if g is not elite]
            del self.pop[min(removable, key=self.score).id]
        a, b = self._tournament(), self._tournament()
        if self.rng.random() < 0.5 and a.id != b.id:
            weights = [x if self.rng.random() < 0.5 else y
                       for x, y in zip(a.weights, b.weights)]
            genes = {k: (a.genes[k] if self.rng.random() < 0.5 else b.genes[k])
                     for k in a.genes}
            parents = [a.id, b.id]
        else:
            weights, genes, parents = list(a.weights), dict(a.genes), [a.id]
        child = Genome(self._new_id(), weights, genes, parents)
        self._mutate(child)
        self.pop[child.id] = child
        return child

    # routing ------------------------------------------------------------

    def pick(self) -> Genome:
        need = [g for g in self.pop.values() if g.id != 0
                and g.jobs() + g.in_flight < self.args.min_jobs]
        if need:
            return min(need, key=lambda g: g.jobs() + g.in_flight)
        if self.rng.random() >= self.args.reeval:
            child = self.breed()
            if child is not None:
                return child
        top = sorted(self.evaluated(), key=lambda g: -self.fit(g))[:3]
        return self.rng.choice(top or [g for g in self.pop.values() if g.id != 0])

    def job(self) -> dict:
        a = self.args
        g = self.pick()
        g.in_flight += 1
        kind = "bust" if self.rng.random() < a.bust_share else "paired"
        if kind == "bust" or not self.hof or self.rng.random() >= a.hof_share:
            if kind == "bust":
                name = self.rng.choice(DUMMIES)
            else:
                name = self.rng.choices(list(DUMMY_WEIGHTS),
                                        weights=list(DUMMY_WEIGHTS.values()))[0]
            opp = {"kind": "dummy", "name": name}
            label = opp["name"]
        else:
            h = self.rng.choice(self.hof)
            opp = {"kind": "genome", "id": h.id, "weights": h.weights,
                   "genes": h.genes}
            label = "hof"
        block = self.rng.randrange(a.blocks)
        seed = 1000 + block
        key = f"{kind}|{label if label != 'hof' else 'hof' + str(opp['id'])}|{block}"
        job = {"key": key, "id": g.id, "type": kind, "opp": opp,
               "opp_label": label, "seed": seed, "weights": g.weights,
               "genes": g.genes, "decks": a.decks, "hands": a.hands}
        if key not in self.base_cache and key not in self.pending:
            base = self.pop[0]
            job["base"] = {"weights": base.weights, "genes": base.genes}
            self.pending[key] = []
        return job

    def record(self, res: dict):
        self.jobs_done += 1
        self.hands += (2 * self.args.decks if res["type"] == "paired"
                       else self.args.hands) * (2 if "base" in res else 1)
        key = res["job"]
        if "base" in res:
            self.base_cache[key] = res["base"]
            waiting = self.pending.pop(key, [])
        else:
            waiting = []
        for r in [res] + waiting:
            if key not in self.base_cache:  # genome 0 still running this key
                self.pending.setdefault(key, []).append(r)
                continue
            g = self.pop.get(r["id"])
            if g is None:
                continue
            g.in_flight -= 1
            diff = r["value"] - self.base_cache[key]
            if r["type"] == "paired":
                g.paired.setdefault(r["opp"], []).append(diff)
            else:
                g.bust_diff.append(float(diff))

    def status(self, start: float):
        mins = (time.time() - start) / 60
        ranked = sorted(self.evaluated(), key=lambda g: -self.score(g))
        print(f"[{mins:5.1f} min] jobs {self.jobs_done} "
              f"({self.jobs_done / max(mins, 1e-9):.0f}/min), hands {self.hands} "
              f"({self.hands / max(mins, 1e-9):.0f}/min), pop {len(self.pop)} "
              f"(next id {self.next_id}), hof {len(self.hof)}, "
              f"cached baselines {len(self.base_cache)}", flush=True)
        print(f"    {'genome':>7} {'score':>7} {'fitness':>8} {'vs #0 bb/100':>16} "
              f"{'jobs':>5} {'bust diff':>9} | per opponent vs #0 (bb/100)",
              flush=True)
        for g in ranked[:5]:
            d = g.diffs()
            mean = sum(d) / len(d)
            per = " ".join(f"{k} {sum(v) / len(v):+.0f}" for k, v in sorted(g.paired.items()))
            bust = (f"{sum(g.bust_diff) / len(g.bust_diff):+.2f}"
                    if g.bust_diff else "-")
            print(f"    {'#' + str(g.id):>7} {self.score(g):>+7.1f} {self.fit(g):>+8.1f} "
                  f"{mean:>+8.1f} ±{1.96 * g.stderr():>5.1f} {g.jobs():>5} "
                  f"{bust:>9} | {per}", flush=True)
        if not ranked:
            print("    (no genome has its first jobs yet)", flush=True)


def run(args):
    evo = Evolution(args)
    if args.resume and os.path.exists(os.path.join(OUT, "pop.json")):
        evo.load()
        print(f"resumed: {len(evo.pop)} genomes, hof {len(evo.hof)}", flush=True)
    else:
        evo.seed_population()
    start = time.time()
    deadline = start + args.minutes * 60
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.append(True))
    last_status = last_save = last_hof = start
    workers = args.workers or os.cpu_count()
    with ProcessPoolExecutor(workers, initializer=_init_worker) as pool:
        running = {pool.submit(play_job, evo.job()) for _ in range(workers * 2)}
        while running:
            done, running = wait(running, timeout=5, return_when=FIRST_COMPLETED)
            for fut in done:
                evo.record(fut.result())
                if time.time() < deadline and not stop:
                    running.add(pool.submit(play_job, evo.job()))
            if stop:
                evo.save()
                evo.status(start)
                print("stopped: saved", os.path.join(OUT, "pop.json"), flush=True)
                for fut in running:
                    fut.cancel()
                for proc in list(pool._processes.values()):
                    proc.kill()
                pool.shutdown(wait=False, cancel_futures=True)
                os._exit(0)
            now = time.time()
            if now - last_status >= args.status_every:
                evo.status(start)
                last_status = now
            if now - last_save >= 60:
                evo.save()
                last_save = now
            if now - last_hof >= args.hof_every * 60:
                ranked = [g for g in evo.evaluated()
                          if g.jobs() >= 2 * args.min_jobs]
                if ranked:
                    evo.to_hof(max(ranked, key=evo.score))
                last_hof = now
    evo.save()
    evo.status(start)
    print("saved", os.path.join(OUT, "pop.json"), flush=True)


def head_to_head(args):
    """Continued games of the top genomes and genome 0 against each dummy,
    same seeds for all: busts and mean chips."""
    with open(os.path.join(OUT, "pop.json")) as f:
        pop = [Genome.from_json(d) for d in json.load(f)["pop"]]
    base = next(g for g in pop if g.id == 0)
    done = [g for g in pop if g.id != 0 and g.jobs() >= args.min_jobs]
    top = sorted(done, key=lambda g: -g.fitness(args.bust_weight))[:args.top]
    jobs = [{"key": f"h2h|{d}|{i}", "id": g.id, "type": "bust",
             "opp": {"kind": "dummy", "name": d}, "opp_label": d,
             "seed": 100_000 + i, "weights": g.weights, "genes": g.genes,
             "decks": 0, "hands": args.hands}
            for g in [base] + top for d in DUMMIES for i in range(args.games)]
    print(f"head to head: genomes {[g.id for g in top]} and #0, "
          f"{args.games} continued games against each dummy", flush=True)
    busts: dict[tuple[int, str], int] = {}
    with ProcessPoolExecutor(args.workers or os.cpu_count(),
                             initializer=_init_worker) as pool:
        for n, res in enumerate(pool.map(play_job, jobs, chunksize=1), 1):
            k = (res["id"], res["opp"])
            busts[k] = busts.get(k, 0) + bool(res["value"])
            if n % max(1, len(jobs) // 10) == 0:
                print(f"  {n}/{len(jobs)} games", flush=True)
    print(f"\n{'genome':>8} " + " ".join(f"{d:>7}" for d in DUMMIES)
          + f" {'total':>7}   (busts out of {args.games} per dummy)")
    for g in [base] + top:
        row = [busts.get((g.id, d), 0) for d in DUMMIES]
        print(f"{'#' + str(g.id):>8} " + " ".join(f"{b:>7}" for b in row)
              + f" {sum(row):>7}")


def export(which: str, args):
    with open(os.path.join(OUT, "pop.json")) as f:
        pop = [Genome.from_json(d) for d in json.load(f)["pop"]]
    pool = [g for g in pop if g.id != 0 and g.jobs() >= args.min_jobs]
    if which == "best":
        g = max(pool, key=lambda g: g.fitness(args.bust_weight) - g.stderr())
    else:
        g = next(g for g in pop if g.id == int(which))
    d = g.diffs()
    mean, se = sum(d) / len(d), g.stderr()
    print(f"genome {g.id}: {mean:+.1f} ± {1.96 * se:.1f} bb/100 vs #0 over "
          f"{len(d)} paired jobs, fitness {g.fitness(args.bust_weight):+.1f}")
    if mean - 1.96 * se <= 0 and not args.force:
        print("not exported: not significantly better than #0 (use --force)")
        return
    path = os.path.join(ROOT, "policy_weights.py")
    with open(path, "w") as f:
        f.write('"""Generated by scripts/evolve.py --export: do not edit by hand.\n\n'
                f"Genome {g.id}: {mean:+.1f} ± {1.96 * se:.1f} bb/100 against the "
                f"plain engine over {len(d)} paired jobs.\n" + '"""\n\n')
        f.write(f"GENES = {json.dumps(g.genes)}\n\n")
        f.write("WEIGHTS = [\n")
        for i in range(0, len(g.weights), 8):
            f.write("    " + ", ".join(f"{w:.6f}" for w in g.weights[i:i + 8]) + ",\n")
        f.write("]\n")
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--export", metavar="ID|best")
    p.add_argument("--force", action="store_true")
    p.add_argument("--h2h", action="store_true")
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--min-jobs", type=int, default=4)  # paired jobs before ranking
    p.add_argument("--decks", type=int, default=50)  # per paired job (x2 hands)
    p.add_argument("--blocks", type=int, default=64)  # deck blocks in the pool
    p.add_argument("--hands", type=int, default=80)  # per bust job
    p.add_argument("--bust-share", type=float, default=0.2)
    p.add_argument("--bust-weight", type=float, default=100.0)  # bb/100 per bust rate
    p.add_argument("--reeval", type=float, default=0.3)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--mut-rate", type=float, default=0.5)
    p.add_argument("--freeze-hidden", type=int, default=1)
    p.add_argument("--status-every", type=float, default=30)  # seconds
    p.add_argument("--hof-every", type=float, default=2)  # minutes
    p.add_argument("--hof-size", type=int, default=10)  # entries kept
    p.add_argument("--hof-share", type=float, default=0.5)  # of paired jobs
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=OUT)  # state folder
    args = p.parse_args()
    globals()["OUT"] = os.path.abspath(args.out)
    if args.export:
        export(args.export, args)
    elif args.h2h:
        head_to_head(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
