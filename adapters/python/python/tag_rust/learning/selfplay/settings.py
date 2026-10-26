"""Versioned self-play configuration; MCTS settings use the existing Rust parser."""
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import math
from tag_rust import _engine
from tag_rust.engine.api import registered_games


def integer(value, name, *, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def number(value, name, *, positive=False):
    try:
        finite = isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or positive and value == 0:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def strict(cls, value):
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
        raise ValueError(f"invalid or unknown {cls.__name__} fields")
    return cls(**value)


def read_json(path):
    path = Path(path)
    try:
        if path.stat().st_size > 1024 * 1024:
            raise ValueError(f"configuration file '{path}' exceeds 1 MiB")
        # Windows PowerShell's UTF-8 output may include a byte-order mark.
        # utf-8-sig accepts ordinary UTF-8 and removes that optional prefix.
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except OSError as error:
        raise ValueError(f"could not read configuration file '{path}': {error}") from error


def neural_search():
    return {"schema_version":1,"tree_policy":"puct","rollout_length":0,
            "budgets":{"iterations":64},"state_evaluator":{"name":"network_value"},
            "puct_preferences":{"source":"probabilities","evaluator":{"name":"network_policy"}}}


def native_search():
    return {"schema_version":1,"rollout_length":0,"budgets":{"iterations":64}}


@dataclass(frozen=True)
class ModelSettings:
    payloads: tuple[str, ...] = ("cells",)
    hidden_sizes: tuple[int, ...] = (64,64)
    value_output: str = "observer"
    value_players: tuple[int,...] | None = None
    value_binding: str = "network_value"
    policy_binding: str = "network_policy"
    value_observers: str | None = None

    def __post_init__(self):
        for name in ("payloads","hidden_sizes"):
            value = getattr(self,name)
            if not isinstance(value,(list,tuple)) or not value:
                raise ValueError(f"{name} must be a nonempty sequence")
            object.__setattr__(self,name,tuple(value))
        if any(not isinstance(n,str) or not n for n in self.payloads) or len(set(self.payloads)) != len(self.payloads):
            raise ValueError("payloads must contain distinct nonempty names")
        for width in self.hidden_sizes:
            integer(width,"hidden layer width")
        if self.value_output not in ("observer","players"):
            raise ValueError("value_output must be observer or players")
        if self.value_players is not None:
            if self.value_output != "players" or not isinstance(self.value_players,(list,tuple)) or not self.value_players:
                raise ValueError("value_players requires a full player-vector output")
            for player in self.value_players:
                integer(player,"value player",minimum=0)
            if len(set(self.value_players)) != len(self.value_players):
                raise ValueError("value_players must be distinct")
            object.__setattr__(self,"value_players",tuple(self.value_players))
        if self.value_observers not in (None,"all_players","root_player","deciding_player"):
            raise ValueError("invalid value_observers")
        if self.value_output == "players" and self.value_observers not in (None,"deciding_player"):
            raise ValueError("full-vector training uses acting-player observations; inference must use deciding_player")
        if any(not isinstance(n,str) or not n or n.startswith("__") for n in (self.value_binding,self.policy_binding)) or self.value_binding == self.policy_binding:
            raise ValueError("use distinct, nonempty value/policy binding names without __ prefixes")


@dataclass(frozen=True)
class ExplorationSettings:
    move_policy: str = "visit_sampling"
    temperature: float = 1.0
    sampling_moves: int | None = 10
    root_noise: bool = True
    noise_fraction: float = 0.25
    noise_alpha: float = 0.3

    def __post_init__(self):
        if self.move_policy not in ("visit_sampling","most_visits"):
            raise ValueError("move_policy must be visit_sampling or most_visits")
        number(self.temperature,"temperature",positive=True)
        if self.sampling_moves is not None:
            integer(self.sampling_moves,"sampling_moves",minimum=0)
        if type(self.root_noise) is not bool:
            raise ValueError("root_noise must be boolean")
        number(self.noise_fraction,"noise_fraction")
        if self.noise_fraction > 1:
            raise ValueError("noise_fraction must be in [0,1]")
        number(self.noise_alpha,"noise_alpha",positive=True)


@dataclass(frozen=True)
class DataSettings:
    mode: str = "replay_buffer"
    capacity: int = 10000
    max_bytes: int = 256 * 1024 * 1024

    def __post_init__(self):
        if self.mode not in ("replay_buffer","newest_group"):
            raise ValueError("data.mode must be replay_buffer or newest_group")
        integer(self.capacity,"data.capacity")
        integer(self.max_bytes,"data.max_bytes")


@dataclass(frozen=True)
class OptimizationSettings:
    updates_per_cycle: int = 64
    batch_size: int = 64
    learning_rate: float = 0.001
    policy_weight: float = 1.0
    value_weight: float = 1.0

    def __post_init__(self):
        integer(self.updates_per_cycle,"updates_per_cycle")
        integer(self.batch_size,"batch_size")
        number(self.learning_rate,"learning_rate",positive=True)
        number(self.policy_weight,"policy_weight")
        number(self.value_weight,"value_weight")
        if self.policy_weight + self.value_weight == 0:
            raise ValueError("enable at least one training loss")


@dataclass(frozen=True)
class EvaluationSettings:
    enabled: bool = True
    games: int = 20
    positions: str = "rotate"
    candidate_seat: int | None = None
    opponent_search: dict = field(default_factory=native_search)
    opponents: tuple[dict, ...] | None = None
    opponent: dict | None = None
    ranking: str | None = None
    max_actions: int | None = None
    include_actions: bool = False

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("evaluation.enabled must be boolean")
        integer(self.games,"evaluation.games")
        if self.positions not in ("rotate","fixed"):
            raise ValueError("evaluation.positions must be rotate or fixed")
        if self.positions == "fixed":
            integer(self.candidate_seat,"candidate_seat",minimum=0)
        elif self.candidate_seat is not None:
            raise ValueError("omit candidate_seat for rotating evaluation")
        value = json.loads(_engine.parse_mcts_settings(json.dumps(self.opponent_search,allow_nan=False)))
        object.__setattr__(self,"opponent_search",value)
        if self.opponents is not None and self.opponent is not None:
            raise ValueError("use opponents or opponent, not both")
        if self.opponents is not None:
            if not isinstance(self.opponents,(list,tuple)) or not self.opponents:
                raise ValueError("opponents must be a nonempty list")
            object.__setattr__(self,"opponents",tuple(_opponent(v) for v in self.opponents))
            names = [v["name"] for v in self.opponents]
            if len(set(names)) != len(names):
                raise ValueError("opponent names must be unique")
        if self.opponent is not None:
            object.__setattr__(self,"opponent",_opponent(self.opponent))
        if (self.opponents is not None or self.opponent is not None) and value != json.loads(
            _engine.parse_mcts_settings(json.dumps(native_search()))
        ):
            raise ValueError("explicit opponents cannot be combined with opponent_search")
        if self.ranking not in (None,"score","claimed_columns"):
            raise ValueError("ranking must be null or score")
        if self.ranking == "claimed_columns":
            object.__setattr__(self,"ranking","score")
        if self.max_actions is not None:
            integer(self.max_actions,"evaluation.max_actions")
        if type(self.include_actions) is not bool:
            raise ValueError("include_actions must be boolean")


def _opponent(value):
    if not isinstance(value,dict) or set(value)!={"name","agent"}:
        raise ValueError("opponent requires name and agent")
    name,agent = value["name"],value["agent"]
    if not isinstance(name,str) or not name.strip() or name == "candidate":
        raise ValueError("use a nonempty opponent name other than candidate")
    if not isinstance(agent,dict) or agent.get("type") not in ("random","osla","mcts"):
        raise ValueError("native opponents support random, osla and mcts")
    allowed = {"random":{"type"},"osla":{"type","repetitions_per_action"},"mcts":{"type","config"}}[agent["type"]]
    if set(agent)-allowed:
        raise ValueError("unknown native opponent setting")
    agent = json.loads(json.dumps(agent,allow_nan=False))
    if agent["type"] == "osla":
        count = integer(agent.get("repetitions_per_action",1),"repetitions_per_action")
        if count > 2**32-1:
            raise ValueError("repetitions_per_action must fit u32")
        agent["repetitions_per_action"] = count
    if agent["type"] == "mcts":
        agent["config"] = json.loads(_engine.parse_mcts_settings(json.dumps(agent.get("config",native_search()),allow_nan=False)))
    return {"name":name,"agent":agent}


@dataclass(frozen=True)
class LimitSettings:
    max_moves: int | None = 512
    max_cells: int = 4096
    max_collection_positions: int = 100000
    max_collection_bytes: int = 256 * 1024 * 1024
    max_group_rows: int = 4096
    max_request_bytes: int = 64 * 1024 * 1024
    max_checkpoint_bytes: int = 512 * 1024 * 1024
    max_model_parameters: int = 10000000
    threads: int = 1

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self,item.name)
            if item.name == "max_moves" and value is None:
                continue
            integer(value,item.name)
        if self.max_model_parameters > 10000000:
            raise ValueError("max_model_parameters exceeds the portable model codec's 10000000-parameter limit")


@dataclass(frozen=True)
class OutputSettings:
    model_file: str | None = None
    checkpoint_file: str | None = None
    report_file: str | None = None

    def __post_init__(self):
        paths = []
        for item in fields(self):
            value = getattr(self,item.name)
            if value is not None:
                if not isinstance(value,str) or not value.strip():
                    raise ValueError(f"{item.name} must be a nonempty path or null")
                paths.append(Path(value).resolve())
        if len(set(paths)) != len(paths):
            raise ValueError("model, checkpoint and report require distinct destinations")


@dataclass(frozen=True)
class SelfPlaySettings:
    schema_version: int = 1
    game: str = "tictactoe"
    parameters: dict = field(default_factory=dict)
    players: int = 2
    seed: int | None = None
    cycles: int = 10
    games_per_cycle: int = 16
    parallel_games: int = 4
    search: dict = field(default_factory=neural_search)
    model: ModelSettings = field(default_factory=ModelSettings)
    exploration: ExplorationSettings = field(default_factory=ExplorationSettings)
    data: DataSettings = field(default_factory=DataSettings)
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    limits: LimitSettings = field(default_factory=LimitSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("self-play schema_version must be 1")
        if self.game not in registered_games():
            raise ValueError(f"unknown game {self.game!r}; registered games: {', '.join(registered_games())}")
        if not isinstance(self.parameters,dict):
            raise ValueError("parameters must be an object")
        for name in ("players","cycles","games_per_cycle","parallel_games"):
            integer(getattr(self,name),name)
        if self.parallel_games > 64:
            raise ValueError("parallel_games exceeds the 64-search coordinator limit")
        if self.seed is not None:
            integer(self.seed,"seed",minimum=0)
            if self.seed >= 2**64:
                raise ValueError("seed exceeds u64")
        for name,cls in (("model",ModelSettings),("exploration",ExplorationSettings),("data",DataSettings),
                         ("optimization",OptimizationSettings),("evaluation",EvaluationSettings),
                         ("limits",LimitSettings),("output",OutputSettings)):
            object.__setattr__(self,name,strict(cls,getattr(self,name)))
        value = json.loads(_engine.parse_mcts_settings(json.dumps(self.search,allow_nan=False)))
        object.__setattr__(self,"search",value)
        if self.exploration.root_noise and (value["tree_policy"] != "puct" or
            value["puct_preferences"].get("source") != "probabilities" or
            value["puct_preferences"].get("evaluator",{}).get("name") != self.model.policy_binding):
            raise ValueError("root noise requires PUCT probabilities using the declared neural policy binding")
        if self.evaluation.candidate_seat is not None and self.evaluation.candidate_seat >= self.players:
            raise ValueError("candidate_seat is outside the game")
        if self.game in ("cantstop", "loveletter"):
            game_name = "Love Letter" if self.game == "loveletter" else "Can't Stop"
            if "seed" in self.parameters:
                raise ValueError(f"{game_name} self-play derives each game seed from the master seed; omit parameters.seed")
            if self.model.value_output != "players" or (
                self.model.value_players is not None and self.model.value_players != tuple(range(self.players))
            ):
                raise ValueError(f"{game_name} self-play requires full absolute-seat player values")
        if self.game == "loveletter":
            if self.players not in (2, 3, 4):
                raise ValueError("Love Letter self-play requires 2–4 players")
            if "last_private_history" in self.model.payloads:
                raise ValueError("Love Letter search cannot reconstruct previous-round private history for hypothetical opponents")
            if value["value_completion"] != "require_full_vector":
                raise ValueError("Love Letter self-play requires require_full_vector completion")
        if self.game in ("cantstop", "loveletter"):
            if self.evaluation.opponents is not None and len(self.evaluation.opponents) != self.players-1:
                raise ValueError("opponents must contain one entry for each non-candidate seat")
        elif (self.evaluation.opponents is not None or self.evaluation.opponent is not None
              or self.evaluation.ranking is not None or self.evaluation.max_actions is not None
              or self.evaluation.include_actions):
            raise ValueError("explicit native lineups and score ranking currently support Can't Stop and Love Letter")

    @classmethod
    def from_dict(cls, value, *, directory=Path(".")):
        if not isinstance(value,dict):
            raise ValueError("self-play configuration must be an object")
        value = json.loads(json.dumps(value,allow_nan=False))
        if "schema_version" not in value:
            raise ValueError("schema_version is required")
        if "search_config_file" in value:
            if "search" in value:
                raise ValueError("use search or search_config_file, not both")
            reference = value.pop("search_config_file")
            if not isinstance(reference,str) or not reference.strip():
                raise ValueError("search_config_file must name a file")
            value["search"] = read_json(directory / reference)
        evaluation = value.get("evaluation",{})
        if isinstance(evaluation,dict):
            if (evaluation.get("opponents") is not None or evaluation.get("opponent") is not None) and any(
                name in evaluation for name in ("opponent_search","opponent_config_file")
            ):
                raise ValueError("explicit opponents cannot be combined with opponent_search")
            entries = evaluation.get("opponents") or ([evaluation["opponent"]] if evaluation.get("opponent") else [])
            for entry in entries:
                if not isinstance(entry,dict) or not isinstance(entry.get("agent"),dict):
                    continue  # Structured validation supplies the configuration error.
                agent = entry["agent"]
                if "config_file" in agent:
                    reference = agent.pop("config_file")
                    if "config" in agent or not isinstance(reference,str) or not reference.strip():
                        raise ValueError("use one valid opponent config or config_file")
                    agent["config"] = read_json(directory / reference)
        if isinstance(evaluation,dict) and "opponent_config_file" in evaluation:
            if "opponent_search" in evaluation:
                raise ValueError("use opponent_search or opponent_config_file, not both")
            reference = evaluation.pop("opponent_config_file")
            if not isinstance(reference,str) or not reference.strip():
                raise ValueError("opponent_config_file must name a file")
            evaluation["opponent_search"] = read_json(directory / reference)
        output = value.get("output",{})
        if isinstance(output,dict):
            for name,path in output.items():
                if isinstance(path,str):
                    output[name] = str((directory / path).resolve())
        return strict(cls,value)

    @classmethod
    def load(cls,path):
        path = Path(path)
        return cls.from_dict(read_json(path),directory=path.parent)

    def document(self):
        value = asdict(self)
        if self.evaluation.opponents is not None or self.evaluation.opponent is not None:
            value["evaluation"].pop("opponent_search")
        return value
