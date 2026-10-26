"""Frozen candidate evaluation with persistent Rust-owned native opponents."""
import json
import time
from tag_rust import _engine
from tag_rust.engine.api import create
from tag_rust.search.mcts import MctsBatch
from .collection import move_settings,start_search,neural_evaluator
from .seeds import workflow_seed,game_parameters
from ..outcomes import RESULT_LABELS,terminal_rewards


def opponents(settings):
    evaluation = settings.evaluation
    if evaluation.opponents is not None:
        return list(evaluation.opponents)
    if evaluation.opponent is not None:
        return [{"name":f"{evaluation.opponent['name']}-{index+1}","agent":evaluation.opponent["agent"]}
                for index in range(settings.players-1)]
    return [{"name":f"native-mcts-{index+1}","agent":{"type":"mcts","config":evaluation.opponent_search}}
            for index in range(settings.players-1)]


def lineup(settings,logical,candidate):
    declared = iter(opponents(settings))
    result = []
    for seat in range(settings.players):
        if seat == candidate:
            result.append({"name":"candidate","agent":{"type":"neural_mcts"}})
            continue
        entry = next(declared)
        seed = workflow_seed(settings,"evaluation",logical,purpose=f"opponent:{entry['name']}")
        agent = json.loads(json.dumps(entry["agent"],allow_nan=False))
        if agent["type"] == "mcts":
            agent["config"]["seed"] = seed
            agent["config"]["final_selection"] = {"policy":"most_visits"}
        result.append({"name":entry["name"],"agent":agent,"seed":seed})
    return result


def native_lineup(game,candidate,actual):
    agents = [None if seat==candidate else dict(entry["agent"],seed=entry["seed"])
              for seat,entry in enumerate(actual)]
    return _engine.NativeOpponents(game._native,candidate,json.dumps(agents,allow_nan=False))


def validate_lineup(settings,snapshot,game):
    """Reject incompatible models and native configurations before scheduled games."""
    if len(opponents(settings)) != settings.players-1:
        raise ValueError("one opponent is required for each non-candidate seat")
    configuration = move_settings(settings,0,0,phase="evaluation",evaluation=True)
    with start_search(game,settings,configuration,neural_evaluator(snapshot)):
        pass
    native_lineup(game,0,lineup(settings,0,0))


def finishing_positions(scores):
    """Higher scores occupy earlier places; a tie averages its occupied places."""
    return [1+sum(other>score for other in scores)+(sum(other==score for other in scores)-1)/2
            for score in scores]


def summarize(rows,players,ranking):
    per_seat = [{"attempted":0,"games":0,"wins":0,"draws":0,"losses":0,"truncated":0,"failed":0}
                for _ in range(players)]
    place_sums = [0.]*players
    for row in rows:
        seat = row["candidate_seat"]
        summary = per_seat[seat]
        summary["attempted"] += 1
        if row["status"] == "completed":
            summary["games"] += 1
            summary[RESULT_LABELS[row["outcomes"][seat]]] += 1
            if ranking is not None:
                place_sums[seat] += row["finishing_positions"][seat]
        else:
            summary["truncated" if row["status"]=="truncated" else "failed"] += 1
    combined = {key:sum(row[key] for row in per_seat) for key in per_seat[0]}
    for seat,row in enumerate(per_seat):
        row["win_rate"] = row["wins"]/row["games"] if row["games"] else None
        if ranking is not None:
            row["average_finishing_position"] = place_sums[seat]/row["games"] if row["games"] else None
    combined["win_rate"] = combined["wins"]/combined["games"] if combined["games"] else None
    if ranking is not None:
        combined["average_finishing_position"] = sum(place_sums)/combined["games"] if combined["games"] else None
    return per_seat,combined


def evaluate_lineup(settings,snapshot,cycle):
    if settings.seed is None:
        raise ValueError("resolve the evaluation master seed before scheduling games")
    if type(cycle) is not int or cycle < 0:
        raise ValueError("cycle must be a nonnegative integer")
    evaluation = settings.evaluation
    first = cycle*evaluation.games
    parameters = game_parameters(settings,first,phase="evaluation")
    with create(settings.game,parameters,players=settings.players,max_cells=settings.limits.max_cells,
                max_bytes=settings.limits.max_request_bytes) as game:
        validate_lineup(settings,snapshot,game)
    evaluator = neural_evaluator(snapshot)
    rows,failures = [],[]
    started = time.perf_counter()
    for index in range(evaluation.games):
        logical = first+index
        seat = logical%settings.players if evaluation.positions=="rotate" else evaluation.candidate_seat
        actual = lineup(settings,logical,seat)
        parameters = game_parameters(settings,logical,phase="evaluation")
        row = {"game_id":logical,"candidate_seat":seat,"game_seed":parameters["seed"],"lineup":actual,
               "moves":0,"status":"failed","outcomes":None}
        if settings.game == "loveletter":
            row["first_player"] = parameters["first_player"]
        actions = []
        decisions = []
        try:
            with create(settings.game,parameters,players=settings.players,max_cells=settings.limits.max_cells,
                        max_bytes=settings.limits.max_request_bytes) as game:
                native = native_lineup(game,seat,actual)
                while True:
                    remaining = None if evaluation.max_actions is None else evaluation.max_actions-row["moves"]
                    result = json.loads(native.advance(game._native,remaining,evaluation.include_actions))
                    row["moves"] += result["moves"]
                    if evaluation.include_actions:
                        if len(actions)+len(result["actions"])>100_000:
                            raise MemoryError("evaluation action trace exceeds 100000 entries")
                        actions.extend(result["actions"])
                    if result["reason"] == "failure":
                        raise RuntimeError(result["error"])
                    if result["reason"] in ("finished","action_limit"):
                        row["status"] = "completed" if result["reason"]=="finished" else "truncated"
                        state = json.loads(native.state(game._native))
                        row["scores"] = state["scores"]
                        if row["status"] == "completed":
                            terminal_rewards(state["outcomes"])
                            row["outcomes"] = state["outcomes"]
                            if evaluation.ranking is not None:
                                row["finishing_positions"] = finishing_positions(state["scores"])
                        break
                    configuration = move_settings(settings,logical,row["moves"],phase="evaluation",evaluation=True)
                    if evaluation.include_actions and len(actions)>=100_000:
                        raise MemoryError("evaluation action trace exceeds 100000 entries")
                    with start_search(game,settings,configuration,evaluator) as search:
                        with MctsBatch([search],max_group_rows=settings.limits.max_group_rows,
                                       max_bytes=settings.limits.max_request_bytes) as batch:
                            report, = batch.run()
                            if batch.errors:
                                raise RuntimeError(str(batch.errors))
                        decisions.append({"ply":row["moves"],"seed":report["settings"]["seed"],
                                          "action_id":report["action_id"],"counters":report["counters"],
                                          "fallback":report["fallback"],"stop_reasons":report["stop_reasons"]})
                        event = json.loads(native.apply_candidate(game._native,report["action_id"],search.position))
                    row["moves"] += 1
                    if evaluation.include_actions:
                        actions.append(event)
                        if len(actions)>100_000:
                            raise MemoryError("evaluation action trace exceeds 100000 entries")
        except Exception as error:
            row["status"] = "failed"
            row["error"] = str(error)
            failures.append({"game_id":logical,"candidate_seat":seat,"error":str(error)})
        row["candidate_decisions"] = decisions
        if evaluation.include_actions:
            row["actions"] = actions
        rows.append(row)
        if row["status"] == "failed":
            break
    per_seat,combined = summarize(rows,settings.players,evaluation.ranking)
    return {"enabled":True,"model_identity":snapshot.identity,
        **({"information_access":"privileged-research", "game":settings.game, "players":settings.players,
            "parameters":json.loads(json.dumps(settings.parameters))} if settings.game == "loveletter" else {}),"master_seed":settings.seed,
        "seed_derivation_version":2,"root_noise":False,"move_policy":"most_visits",
        "requested_games":evaluation.games,"attempted_games":len(rows),"completed_games":combined["games"],
        "truncated_games":combined["truncated"],"failed_games":combined["failed"],
        "unplayed_games":evaluation.games-len(rows),"games":rows,"failures":failures,
        "combined":combined,"per_seat":per_seat,"candidate_search":json.loads(json.dumps(settings.search)),
        "opponents":json.loads(json.dumps(opponents(settings))),"ranking":evaluation.ranking,"max_actions":evaluation.max_actions,
        "elapsed_ms":(time.perf_counter()-started)*1000}
