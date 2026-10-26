"""Separate, seat-balanced evaluation of a frozen candidate against native MCTS."""
import time
import json
from tag_rust.engine.api import create
from tag_rust.search.mcts import MctsBatch,MctsSearch
from .collection import move_settings,start_search,terminal_rewards,neural_evaluator
from .seeds import derive_seed
from ..outcomes import RESULT_LABELS, outcome_counts


def evaluate_model(settings,snapshot,*,cycle=0):
    """Evaluate a frozen model independently of training; resolve one optional master seed.

    settings is a SelfPlaySettings instance or its configuration mapping. snapshot
    is a FrozenModel with its declared game/rules/value contract. Returned reports
    contain owned metadata; model weights and training RNGs are unchanged.
    """
    from dataclasses import replace
    from tag_rust import _engine
    from tag_rust.learning.snapshot import FrozenModel
    from .settings import SelfPlaySettings
    if isinstance(settings,dict):
        settings = SelfPlaySettings.from_dict(settings)
    if not isinstance(settings,SelfPlaySettings) or not isinstance(snapshot,FrozenModel):
        raise TypeError("evaluate_model requires SelfPlaySettings and FrozenModel")
    if type(cycle) is not int or cycle<0:
        raise ValueError("cycle must be a nonnegative integer")
    if settings.seed is None:
        settings = replace(settings,seed=_engine.automatic_search_seed())
    result = evaluate(settings,snapshot,cycle)
    result.setdefault("master_seed",settings.seed)
    return json.loads(json.dumps(result,allow_nan=False))


def evaluate(settings,snapshot,cycle):
    if not settings.evaluation.enabled:
        return {"enabled":False}
    if settings.game in ("cantstop", "loveletter"):
        from .lineup_evaluation import evaluate_lineup
        return evaluate_lineup(settings,snapshot,cycle)
    if settings.players != 2:
        raise ValueError("candidate/opponent evaluation currently requires a two-player game")
    evaluator = neural_evaluator(snapshot)
    rows,failures = [],[]
    started = time.perf_counter()
    counts = [outcome_counts(counter="games") for _ in range(settings.players)]
    for index in range(settings.evaluation.games):
        seat = index % settings.players if settings.evaluation.positions == "rotate" else settings.evaluation.candidate_seat
        logical = cycle*settings.evaluation.games+index
        try:
            with create(settings.game,settings.parameters,players=settings.players,max_cells=settings.limits.max_cells,
                        max_bytes=settings.limits.max_request_bytes) as game:
                ply = 0
                while game.next_player is not None:
                    if settings.limits.max_moves is not None and ply >= settings.limits.max_moves:
                        raise RuntimeError("evaluation reached max_moves without terminal outcomes")
                    actor = game.next_player
                    if actor == seat:
                        configuration = move_settings(settings,logical,ply,phase="evaluation",evaluation=True)
                        search = start_search(game,settings,configuration,evaluator)
                    else:
                        configuration = dict(settings.evaluation.opponent_search)
                        configuration["seed"] = derive_seed(settings.seed,"evaluation",logical,ply,"native-opponent")
                        configuration["final_selection"] = {"policy":"most_visits"}
                        search = MctsSearch(game,configuration,max_rows=settings.limits.max_group_rows,
                                            max_bytes=settings.limits.max_request_bytes)
                    with search:
                        with MctsBatch([search],max_group_rows=settings.limits.max_group_rows,
                                       max_bytes=settings.limits.max_request_bytes) as batch:
                            report, = batch.run()
                            if batch.errors:
                                raise RuntimeError(str(batch.errors))
                        game.apply(report["action_id"],position=search.position)
                    ply += 1
                codes = tuple(int(code) for code in game.observe(0,()).results)
                terminal_rewards(codes)
                result = codes[seat]
                counts[seat]["games"] += 1
                counts[seat][RESULT_LABELS[result]] += 1
                rows.append({"game_id":logical,"candidate_seat":seat,"outcomes":list(codes),"moves":ply})
        except Exception as error:
            failures.append({"game_id":logical,"candidate_seat":seat,"error":str(error)})
            break
    combined = {key:sum(row[key] for row in counts) for key in counts[0]}
    for row in (*counts,combined):
        row["win_rate"] = row["wins"]/row["games"] if row["games"] else None
    return {"enabled":True,"model_identity":snapshot.identity,"root_noise":False,"move_policy":"most_visits",
        "requested_games":settings.evaluation.games,"completed_games":len(rows),"games":rows,"failures":failures,
        "combined":combined,"per_seat":counts,"candidate_search":settings.search,
        "opponent_search":settings.evaluation.opponent_search,
        "decision_overrides":{"final_selection":{"policy":"most_visits"},"seed_source":"tag-m8-selfplay-seed/1"},
        "elapsed_ms":(time.perf_counter()-started)*1000}
