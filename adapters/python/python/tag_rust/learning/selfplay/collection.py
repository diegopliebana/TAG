"""Generate complete self-play games while batching independent Rust searches."""
from dataclasses import dataclass,field
import json
import time
import numpy as np
from tag_rust.engine.api import create
from tag_rust.search.mcts import MctsSearch,MctsBatch
from tag_rust.search.neural_mcts import NeuralMctsEvaluator
from .data import PositionExample,owned_observation,visit_target,validate_example
from .noise import RootNoiseEvaluator
from .seeds import workflow_seed,game_parameters
from ..outcomes import terminal_rewards


@dataclass(frozen=True)
class SelfPlayGame:
    game_id: int
    outcomes: tuple[int,...]
    positions: tuple[PositionExample,...]
    game_seed: int | None = None


@dataclass(frozen=True)
class CollectionResult:
    games: tuple[SelfPlayGame,...]
    metrics: dict

    @property
    def positions(self):
        return tuple(position for game in self.games for position in game.positions)


@dataclass
class _ActiveGame:
    logical_id: int
    game: object
    records: list = field(default_factory=list)
    search: object = None
    observations: tuple = ()
    actor: int = 0
    noise_seed: int = 0
    search_ids: set = field(default_factory=set)


def observer_mode(settings):
    return settings.model.value_observers or ("all_players" if settings.model.value_output == "observer" else "deciding_player")


def neural_evaluator(snapshot):
    context = "any_turn" if snapshot.contract.value.players is None else "own_turn"
    return NeuralMctsEvaluator(snapshot,turn_context=context)


def bindings(settings,configuration,evaluator):
    result = {}
    if configuration["state_evaluator"]["name"] == settings.model.value_binding:
        result[settings.model.value_binding] = evaluator
    if configuration["puct_preferences"].get("evaluator",{}).get("name") == settings.model.policy_binding:
        result[settings.model.policy_binding] = evaluator
    return result


def move_settings(settings,game_id,ply,*,phase="selfplay",evaluation=False):
    configuration = json.loads(json.dumps(settings.search))
    configuration["seed"] = workflow_seed(settings,phase,game_id,ply,"search")
    exploration = settings.exploration
    sampling = not evaluation and exploration.move_policy == "visit_sampling" and (
        exploration.sampling_moves is None or ply < exploration.sampling_moves)
    configuration["final_selection"] = ({"policy":"visit_sampling","temperature":exploration.temperature}
                                         if sampling else {"policy":"most_visits"})
    return configuration


def start_search(game,settings,configuration,evaluator):
    return MctsSearch(game,configuration,bindings=bindings(settings,configuration,evaluator),
                      value_observers=observer_mode(settings),max_rows=settings.limits.max_group_rows,
                      max_bytes=settings.limits.max_request_bytes)


def collect_games(settings,snapshot,*,first_game_id=0):
    """A failed/incomplete game aborts this group before any training data is added.

    Records are inserted in logical game/ply order after all games complete,
    independently of which active worker or game finishes first.
    """
    if settings.seed is None:
        raise ValueError("resolve the master seed before collecting self-play")
    contract = snapshot.contract
    evaluator = neural_evaluator(snapshot)
    if settings.exploration.root_noise:
        evaluator = RootNoiseEvaluator(evaluator,settings.exploration)
    active,completed = [],[]
    next_game = first_game_id
    end_game = first_game_id+settings.games_per_cycle
    metrics = {"positions":0,"numeric_bytes":0,"inference_calls":0,"inference_rows":0,
               "max_batch_rows":0,"inference_ms":0.,"search_counters":{}}
    started = time.perf_counter()

    def fill_slots():
        nonlocal next_game
        while len(active) < settings.parallel_games and next_game < end_game:
            game = create(settings.game,game_parameters(settings,next_game),players=settings.players,
                          max_cells=settings.limits.max_cells,max_bytes=settings.limits.max_request_bytes)
            active.append(_ActiveGame(next_game,game))
            next_game += 1

    try:
        fill_slots()
        while active:
            for member in active:
                if member.search is None:
                    member.actor = member.game.next_player
                    if member.actor is None:
                        raise ValueError("self-play initialization produced no decision")
                    ply = len(member.records)
                    if settings.limits.max_moves is not None and ply >= settings.limits.max_moves:
                        raise RuntimeError(f"game {member.logical_id} reached max_moves without terminal outcomes")
                    observers = range(settings.players) if contract.value.players is None else (member.actor,)
                    member.observations = tuple(owned_observation(member.game.observe(player,settings.model.payloads)) for player in observers)
                    configuration = move_settings(settings,member.logical_id,ply)
                    member.noise_seed = workflow_seed(settings,"selfplay",member.logical_id,ply,"root-noise")
                    member.search = start_search(member.game,settings,configuration,evaluator)
                    member.search_ids.clear()
            # The coordinator does not own these search lifetimes: completed
            # decisions close themselves, and the finally block cancels failures.
            batch = MctsBatch([member.search for member in active],
                              max_group_rows=settings.limits.max_group_rows,max_bytes=settings.limits.max_request_bytes)
            groups = batch.advance()
            if batch.errors:
                raise RuntimeError(f"self-play searches failed: {batch.errors}")
            for group in groups:
                requests = []
                for index,request in group.members:
                    member = active[index]
                    member.search_ids.add(request.id[0])
                    if isinstance(evaluator,RootNoiseEvaluator):
                        evaluator.bind(request,member.noise_seed)
                    requests.append(request)
                before = time.perf_counter()
                replies = group.evaluator.evaluate(requests,max_bytes=settings.limits.max_request_bytes)
                metrics["inference_ms"] += (time.perf_counter()-before)*1000
                metrics["inference_calls"] += 1
                metrics["inference_rows"] += group.rows
                metrics["max_batch_rows"] = max(metrics["max_batch_rows"],group.rows)
                batch.resume(group,replies)
            if batch.errors:
                raise RuntimeError(f"self-play searches failed: {batch.errors}")
            for member in tuple(active):
                if member.search.report is None:
                    continue
                report = member.search.report
                actor_observation = next(obs for obs in member.observations if obs.observer == member.actor)
                visits,policy = visit_target(actor_observation,report)
                metadata = {name:report[name] for name in ("settings","counters","action_id","stop_reasons",
                    "fallback","nodes","edges","external_evaluators","value_observers")}
                record = dict(observations=member.observations,actor=member.actor,policy=policy,visits=visits,
                    action=report["action_id"],game_id=member.logical_id,ply=len(member.records),
                    model_identity=snapshot.identity,search_seed=report["settings"]["seed"],
                    noise_seed=member.noise_seed,search_report=metadata)
                # Reserve terminal-vector bytes before retaining an unfinished
                # position. The placeholder never enters the training buffer.
                pending = PositionExample(**record,terminal_values=np.zeros(settings.players,dtype=np.float32))
                if metrics["positions"]+1 > settings.limits.max_collection_positions or metrics["numeric_bytes"]+pending.nbytes > settings.limits.max_collection_bytes:
                    raise MemoryError("self-play group exceeds collection storage limits")
                metrics["positions"] += 1
                metrics["numeric_bytes"] += pending.nbytes
                for name,value in report["counters"].items():
                    metrics["search_counters"][name] = metrics["search_counters"].get(name,0)+value
                member.records.append(record)
                member.game.apply(report["action_id"],position=member.search.position)
                if isinstance(evaluator,RootNoiseEvaluator):
                    evaluator.release(member.search_ids)
                member.search = None
                if member.game.next_player is None:
                    codes = tuple(int(code) for code in member.game.observe(0,()).results)
                    rewards = terminal_rewards(codes)
                    positions = tuple(PositionExample(**record,terminal_values=rewards.copy()) for record in member.records)
                    for position in positions:
                        validate_example(position,contract,max_bytes=settings.data.max_bytes)
                    completed.append(SelfPlayGame(member.logical_id,codes,positions,member.game.seed))
                    member.game.close()
                    active.remove(member)
            fill_slots()
    finally:
        for member in active:
            if member.search is not None:
                member.search.close()
            member.game.close()
    metrics["generation_ms"] = (time.perf_counter()-started)*1000
    return CollectionResult(tuple(sorted(completed,key=lambda game:game.game_id)),metrics)
