"""Portable models and completed-cycle checkpoints containing bounded training data."""
from pathlib import Path
import json
import numpy as np
import torch
from ..checkpoint_state import restore_adam_groups, validate_adam_tensors, restored_torch_rng
from tag_rust.engine.api import Observation,create
from tag_rust.learning.artifacts import _atomic_write,model_document,load_document,save_model
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.training import runtime_identity
from .settings import SelfPlaySettings
from .data import PositionExample,validate_example
from .collection import terminal_rewards
from .seeds import workflow_seed,seed_version,game_parameters
from ..outcomes import seat_outcome_counts


def _observation_document(observation):
    return {"observer":observation.observer,"next_player":observation.next_player,
        "payloads":{name:torch.from_numpy(value.copy()) for name,value in observation.payloads.items()},
        **{name:torch.from_numpy(getattr(observation,name).copy()) for name in
           ("progress","results","action_mask","offered_action_ids")}}


def _position_document(position):
    return {"observations":[_observation_document(obs) for obs in position.observations],
        **{name:torch.from_numpy(getattr(position,name).copy()) for name in ("policy","visits","terminal_values")},
        **{name:getattr(position,name) for name in ("actor","action","game_id","ply","model_identity",
                                                    "search_seed","noise_seed","search_report")}}


def save_checkpoint(path,session):
    session.check_boundary()
    document = {"format":"tag-mcts-selfplay-training","version":seed_version(session.settings),"algorithm_version":1,
        "model":model_document(session.model,provenance=session.provenance()),"settings":session.settings.document(),
        "initial_identity":session.initial_identity,"completed_cycles":session.completed_cycles,
        "next_game_id":session.next_game_id,"updates":session.updates,"history":session.history,
        "optimizer":session.optimizer.state_dict(),"torch_rng":session.torch_rng.clone(),
        "sample_rng":session.sample_rng.bit_generator.state,"runtime":session.runtime,
        "positions":[_position_document(position) for position in session.buffer._positions],
        "buffer_bytes":session.buffer.nbytes,"buffer_dropped":session.buffer.dropped,"buffer_digest":session.buffer.digest()}
    if session.settings.game == "loveletter":
        document["information_access"] = document["model"]["provenance"]["information_access"]
    def write(stream):
        torch.save(document,stream)
        if stream.tell() > session.settings.limits.max_checkpoint_bytes:
            raise ValueError("self-play checkpoint exceeds max_checkpoint_bytes")
    _atomic_write(path,write)


def _array(value):
    if not isinstance(value,torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided:
        raise ValueError("checkpoint arrays must be dense CPU tensors")
    return value.detach().numpy().copy()


def _position(value,contract):
    observations = []
    for row in value["observations"]:
        observations.append(Observation(identity=contract.identity,position=None,observer=row["observer"],
            next_player=row["next_player"],payloads={name:_array(array) for name,array in row["payloads"].items()},
            **{name:_array(row[name]) for name in ("progress","results","action_mask","offered_action_ids")}))
    return PositionExample(observations=tuple(observations),
        **{name:_array(value[name]) for name in ("policy","visits","terminal_values")},
        **{name:value[name] for name in ("actor","action","game_id","ply","model_identity","search_seed","noise_seed","search_report")})


def load_checkpoint(path,*,max_bytes=512*1024*1024):
    """Restore only completed cycles under matching recorded CPU/runtime conditions."""
    from .training import SelfPlaySession
    path = Path(path)
    if type(max_bytes) is not int or max_bytes <= 0 or path.stat().st_size > max_bytes:
        raise ValueError("checkpoint exceeds the requested file-size limit")
    data = torch.load(path,map_location="cpu",weights_only=True)
    if (not isinstance(data,dict) or type(data.get("version")) is not int or type(data.get("algorithm_version")) is not int
        or data.get("format") != "tag-mcts-selfplay-training" or data.get("version") not in (1,2)
        or data.get("algorithm_version") != 1):
        raise ValueError("unsupported self-play training format/version")
    try:
        settings = SelfPlaySettings.from_dict(data["settings"])
        if data["version"] != seed_version(settings):
            raise ValueError("checkpoint format and game seed derivation version differ")
        if settings.seed is None or path.stat().st_size > settings.limits.max_checkpoint_bytes:
            raise ValueError("checkpoint has unresolved seed or exceeds its declared limit")
        current = runtime_identity()
        current.update(threads=settings.limits.threads,deterministic=True)
        if current != data["runtime"]:
            raise ValueError("checkpoint CPU/software runtime differs from this execution")
        completed,updates,next_game = data["completed_cycles"],data["updates"],data["next_game_id"]
        if (type(completed) is not int or not 0 <= completed <= settings.cycles or type(updates) is not int
            or type(next_game) is not int or updates != completed*settings.optimization.updates_per_cycle
            or next_game != completed*settings.games_per_cycle or len(data["history"]) != completed):
            raise ValueError("checkpoint counters differ from completed cycles")
        # Validate the schedule before restoring the model or its optimizer.
        session = SelfPlaySession(settings)
        loaded = load_document(data["model"],expected_contract=session.model.contract)
        if seed_version(settings) == 2:
            provenance = data["model"].get("provenance")
            if not isinstance(provenance,dict) or provenance.get("seed_derivation_version") != 2:
                raise ValueError("checkpoint model seed derivation provenance differs")
            if settings.game == "loveletter" and (
                provenance.get("settings") != settings.document()
                or provenance.get("master_seed") != settings.seed
                or provenance.get("completed_cycles") != completed
                or provenance.get("next_game_id") != next_game
                or provenance.get("updates") != updates
                or provenance.get("information_access") != "privileged-research"
                or data.get("information_access") != provenance.get("information_access")
            ):
                raise ValueError("checkpoint model training provenance differs from its settings/counters")
        identity = FrozenModel(loaded).identity
        previous = data["initial_identity"]
        games = {}
        for index,report in enumerate(data["history"]):
            if (report["cycle"] != index+1 or report["before_identity"] != previous
                or report["first_game_id"] != index*settings.games_per_cycle
                or report["next_game_id"] != (index+1)*settings.games_per_cycle
                or report["updates"] != (index+1)*settings.optimization.updates_per_cycle
                or len(report["games"]) != settings.games_per_cycle
                or len(report["optimization"]) != settings.optimization.updates_per_cycle):
                raise ValueError("checkpoint history is not a completed-cycle schedule")
            for offset,game in enumerate(report["games"]):
                logical = index*settings.games_per_cycle+offset
                if game["game_id"] != logical or type(game["moves"]) is not int or game["moves"] < 1 or len(game["outcomes"]) != settings.players:
                    raise ValueError("invalid completed-game history")
                terminal_rewards(game["outcomes"])
                if settings.game == "loveletter" and (1 not in game["outcomes"] or any(
                    code not in (1,3) for code in game["outcomes"]
                )):
                    raise ValueError("Love Letter completed games require winners and win/loss outcomes")
                if seed_version(settings) == 2 and (
                    type(game.get("game_seed")) is not int or game["game_seed"] !=
                    workflow_seed(settings,"selfplay",logical,purpose="game-chance")
                ):
                    raise ValueError("checkpoint real-game seed provenance differs")
                games[logical] = (game,report["before_identity"])
            if seed_version(settings) == 2:
                summary = seat_outcome_counts((game["outcomes"] for game in report["games"]),settings.players)
                if any(report.get(key) != value for key,value in summary.items()):
                    raise ValueError("checkpoint seat summaries differ from recorded games")
            previous = report["model_identity"]
        if previous != identity:
            raise ValueError("checkpoint weights differ from completed model history")
        if len(data["positions"]) > settings.data.capacity:
            raise ValueError("checkpoint exceeds position capacity")
        positions = []
        prior = (-1,-1)
        for record in data["positions"]:
            position = _position(record,session.model.contract)
            validate_example(position,session.model.contract,max_bytes=settings.data.max_bytes)
            key = (position.game_id,position.ply)
            if key <= prior or position.game_id not in games:
                raise ValueError("checkpoint positions are not ordered completed-game records")
            game,source_identity = games[position.game_id]
            if (position.ply >= game["moves"] or position.model_identity != source_identity
                or not np.array_equal(position.terminal_values,terminal_rewards(game["outcomes"]))
                or position.search_seed != workflow_seed(settings,"selfplay",position.game_id,position.ply,"search")
                or position.noise_seed != workflow_seed(settings,"selfplay",position.game_id,position.ply,"root-noise")):
                raise ValueError("checkpoint position outcome/seed/model provenance differs")
            if settings.data.mode == "newest_group" and position.game_id < (completed-1)*settings.games_per_cycle:
                raise ValueError("newest-group checkpoint retains an earlier cycle")
            positions.append(position)
            prior = key
        session.buffer.add_group(positions)
        if (type(data["buffer_bytes"]) is not int or data["buffer_bytes"] < 0
            or len(session.buffer) != len(positions) or session.buffer.nbytes != data["buffer_bytes"]
            or session.buffer.digest() != data["buffer_digest"] or completed > 0 and not positions
            or type(data["buffer_dropped"]) is not int or data["buffer_dropped"] < 0):
            raise ValueError("checkpoint buffer bytes/digest/counters differ")
        if completed:
            last = data["history"][-1]
            if (last["buffer_digest"] != data["buffer_digest"] or last["buffer_bytes"] != data["buffer_bytes"]
                or last["buffer_positions"] != len(positions) or last["dropped_positions"] != data["buffer_dropped"]):
                raise ValueError("checkpoint buffer differs from its completed cycle")
        elif positions or data["buffer_dropped"] or data["buffer_bytes"]:
            raise ValueError("initial checkpoint must have an empty buffer")
        if settings.game == "loveletter":
            _validate_replay(settings,data["history"],positions)
        # Numeric input arrays are now independently owned and checked. Restore
        # Adam only after its settings agree with this completed-cycle schedule.
        session.model.load_state_dict(loaded.state_dict())
        restore_adam_groups(session.optimizer, data["optimizer"],
                            "checkpoint Adam settings differ from configuration")
        if not updates and session.optimizer.state:
            raise ValueError("initial checkpoint unexpectedly has Adam state")
        if updates and len(session.optimizer.state) != len(list(session.model.parameters())):
            raise ValueError("checkpoint is missing Adam parameter state")
        validate_adam_tensors(session.optimizer, updates,
            state_error="invalid Adam parameter state",
            tensor_error="invalid Adam tensor",
            step_error="Adam update count differs from completed cycles")
        session.torch_rng = restored_torch_rng(data["torch_rng"])
        session.sample_rng.bit_generator.state = data["sample_rng"]
        session.initial_identity = data["initial_identity"]
        session.buffer.dropped = data["buffer_dropped"]
        session.completed_cycles,session.updates,session.next_game_id = completed,updates,next_game
        session._history = data["history"]
        json.dumps(session.history,allow_nan=False)
        session.check_boundary()
        return session
    except (KeyError,TypeError,OverflowError,IndexError) as error:
        raise ValueError("malformed self-play checkpoint") from error


def _validate_replay(settings,history,positions):
    """Verify stored real-game inputs, not just their shapes or saved digest.

    Love Letter records complete action traces even when older examples leave
    the bounded buffer. Rust replays each trace and supplies the permitted view.
    This is privileged checkpoint validation, never a model input path.
    """
    retained = {(p.game_id,p.ply):p for p in positions}
    for cycle in history:
        count = 0
        for record in cycle["games"]:
            parameters = game_parameters(settings,record["game_id"])
            actions = record.get("actions")
            if (type(record.get("first_player")) is not int
                or not isinstance(actions,list) or len(actions) != record["moves"]
                or any(type(action) is not int or action < 0 for action in actions)):
                raise ValueError("checkpoint replay actions/starter differ from recorded game")
            count += len(actions)
            if count > settings.limits.max_collection_positions or (
                settings.limits.max_moves is not None and len(actions) > settings.limits.max_moves
            ):
                raise ValueError("checkpoint replay exceeds collection work limits")
            with create(settings.game,parameters,players=settings.players,max_cells=settings.limits.max_cells,
                        max_bytes=settings.limits.max_request_bytes) as game:
                if record["first_player"] != game.next_player:
                    raise ValueError("checkpoint starter differs from real-game replay")
                for ply,action in enumerate(actions):
                    actor = game.next_player
                    if actor is None:
                        raise ValueError("checkpoint replay continues a finished game")
                    position = retained.get((record["game_id"],ply))
                    if position is not None:
                        if position.actor != actor or position.action != action or len(position.observations) != 1:
                            raise ValueError("checkpoint position differs from real-game replay")
                        actual = game.observe(actor,settings.model.payloads)
                        saved = position.observations[0]
                        if (actual.observer != saved.observer or actual.next_player != saved.next_player
                            or tuple(actual.payloads) != tuple(saved.payloads)
                            or any(not np.array_equal(actual.payloads[name],saved.payloads[name]) for name in actual.payloads)
                            or any(not np.array_equal(getattr(actual,name),getattr(saved,name)) for name in
                                   ("progress","results","action_mask","offered_action_ids"))):
                            raise ValueError("checkpoint observation differs from permitted real-game replay")
                    game.apply(action,position=game.position)
                if game.next_player is not None or game.observe(0,()).results.tolist() != record["outcomes"]:
                    raise ValueError("checkpoint completed outcomes differ from real-game replay")


def save_outputs(session):
    session.check_boundary()
    output = session.settings.output
    if output.model_file is not None:
        save_model(output.model_file,session.model,provenance=session.provenance())
    if output.checkpoint_file is not None:
        save_checkpoint(output.checkpoint_file,session)
    if output.report_file is not None:
        document = {"format":"tag-mcts-selfplay-results","version":1,
                    **session.provenance(),"history":session.history}
        text = json.dumps(document,indent=2,allow_nan=False).encode("utf-8")
        _atomic_write(output.report_file,lambda stream:stream.write(text))
