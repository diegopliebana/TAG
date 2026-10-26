"""Resume Rust MCTS decisions and batch evaluation across independent trees."""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from collections.abc import Mapping
import json
import numpy as np
from tag_rust.engine.api import Game, Observation, EncodingIdentity, _integer
from tag_rust import _engine


@dataclass(frozen=True)
class MctsRequest:
    """One tree's current evaluation, with independently owned observation arrays."""
    id: tuple[int, int]
    kind: str
    binding: str
    evaluator_identity: str
    root_player: int
    deciding_player: int
    observations: tuple[Observation, ...]
    action_ids: tuple[int, ...]
    policy_node: int | None = None

    @property
    def rows(self):
        return len(self.observations)

    @property
    def nbytes(self):
        return sum(sum(a.nbytes for a in obs.payloads.values()) + obs.action_mask.nbytes
                   + obs.offered_action_ids.nbytes + obs.results.nbytes + obs.progress.nbytes
                   for obs in self.observations)


@dataclass(frozen=True)
class MctsResponse:
    """Explicitly identify a response's request, evaluator and legal action order."""
    request_id: tuple[int, int]
    evaluator_identity: str
    action_ids: tuple[int, ...]
    values: np.ndarray
    scalar_player: int | None = None

    @classmethod
    def for_request(cls, request, values, *, scalar_player=None):
        return cls(request.id, request.evaluator_identity, request.action_ids,
                   np.array(values, dtype=np.float64, copy=True), scalar_player)


def _settings(value):
    if isinstance(value, Mapping):
        value = dict(value)
    else:
        text = str(value)
        value = json.loads(text if text.lstrip().startswith(('{', '[')) else Path(value).read_text())
    # Reject NaN/Infinity and copy nested mutable configuration dictionaries.
    return json.loads(_engine.parse_mcts_settings(json.dumps(value, allow_nan=False)))


class MctsSearch:
    """Search a captured position without applying a move to its real game.

    settings accepts an MCTS JSON path, JSON text or mapping. bindings maps state
    and policy evaluator names to external evaluators. Other roles remain native.
    A fresh search owns a fresh agent; seed=None uses Rust's OS randomness.
    """
    def __init__(self, game, settings, *, bindings=None, value_observers=None,
                 max_rows=4096, max_bytes=64 * 1024 * 1024):
        if not isinstance(game, Game):
            raise TypeError("game must be a tag_rust Game")
        self.game, self.position = game, game.position
        self.bindings = dict(bindings or {})
        if any(not isinstance(name, str) or not name or name.startswith("__") for name in self.bindings):
            raise ValueError("binding names must be nonempty strings without a reserved __ prefix")
        configuration = _settings(settings)
        state = configuration.get("state_evaluator", {}).get("name", "game_heuristic")
        policy = configuration.get("puct_preferences", {}).get("evaluator", {}).get("name")
        self._names = {"values": state, "policy": policy}
        if any(name not in (state, policy) for name in self.bindings):
            raise ValueError("external bindings are supported only for state values and PUCT preferences")
        if state == policy and state in self.bindings:
            raise ValueError("use distinct state and policy evaluator names, even for the same model")
        for role in ("progressive_bias", "expansion", "rollout"):
            spec = configuration.get(role, {}).get("evaluator", {})
            if spec.get("name") in self.bindings:
                raise ValueError(f"{role} action heuristics remain native in this stage")
        if policy in self.bindings and (configuration.get("tree_policy", "ucb1") != "puct"
                                       or configuration["puct_preferences"].get("source") != "probabilities"):
            raise ValueError("external PUCT preferences require tree_policy=puct and source=probabilities")
        self.max_rows, self.max_bytes = _integer(max_rows, "max_rows"), _integer(max_bytes, "max_bytes")
        if self.max_rows == 0 or self.max_bytes == 0:
            raise ValueError("row and byte limits must be positive")
        spec = game.spec()
        if value_observers is None:
            value_observers = "deciding_player" if spec.identity.game in ("cantstop", "loveletter") else "all_players"
        if spec.identity.game == "loveletter":
            if (value_observers != "deciding_player"
                    or configuration["value_completion"] != "require_full_vector"):
                raise ValueError("Love Letter MCTS requires deciding_player observations and require_full_vector completion")
        if spec.identity.game == "cantstop" and state in self.bindings:
            if (value_observers != "deciding_player"
                    or configuration["value_completion"] != "require_full_vector"):
                raise ValueError("Can't Stop external values require deciding_player observations and require_full_vector completion")
        names = []
        self._identities = {}
        for kind, name in self._names.items():
            if name not in self.bindings:
                continue
            evaluator = self.bindings[name]
            if not isinstance(evaluator.identity, str) or not evaluator.identity:
                raise ValueError("external evaluator must declare a nonempty identity")
            evaluator.validate(spec, kind, value_observers,
                               configuration.get("value_completion", "require_full_vector"))
            self._identities[name] = evaluator.identity
            for payload in evaluator.payloads:
                if payload not in names:
                    names.append(payload)
        # Validate the requested names before any search work is performed.
        if spec.identity.game == "loveletter" and "last_private_history" in names:
            raise ValueError("Love Letter MCTS cannot supply previous-round private history for hypothetical opponents")
        if game.next_player is None:
            raise ValueError("cannot search a finished game")
        game.observe(game.next_player, names)
        self._native = game._native.start_mcts(
            json.dumps(configuration, allow_nan=False), value_observers, names,
            state if state in self.bindings else "__unused_external_values",
            policy if policy in self.bindings else "__unused_external_policy",
            self.max_rows, self.max_bytes)
        self.settings = json.loads(self._native.settings_json)
        self.value_observers = value_observers
        self.pending = None
        self.report = None
        self.error = None
        self._closed = False

    def _current(self):
        try:
            position = self.game.position
        except _engine.ClosedGameError:
            self.close()
            raise
        if position != self.position:
            self.close()
            raise _engine.StalePositionError("MCTS root game changed after search creation")
        if self._closed and self.report is None:
            raise RuntimeError("search is closed")

    def _consume(self, raw):
        if raw["reason"] == "yielded":
            self.pending = None
            return None
        if raw["reason"] == "finished":
            self.pending = None
            self.report = json.loads(raw["report_json"])
            self.report["external_evaluators"] = dict(self._identities)
            self.report["value_observers"] = self.value_observers
            self.close()
            return self.report
        if "request" in raw:
            raw = raw["request"]
            name = self._names[raw["kind"]]
            if name not in self.bindings:
                self.close()
                raise ValueError("no external evaluator bound for request")
            rows = []
            for row in raw["observations"]:
                row["identity"] = EncodingIdentity(**row["identity"])
                row["position"] = self.position
                row["payloads"] = MappingProxyType(row["payloads"])
                rows.append(Observation(**row))
            self.pending = MctsRequest((raw["search_id"], raw["sequence"]), raw["kind"], name,
                self._identities[name], raw["root_player"], raw["deciding_player"],
                tuple(rows), tuple(raw["action_ids"]), raw["policy_node"])
        return self.pending

    def advance(self):
        """Continue until an evaluation or completion; also poll deadlines while waiting."""
        self._current()
        if self.report is not None:
            return self.report
        try:
            return self._consume(self._native.advance())
        except Exception as error:
            self.error = str(error)
            self.close()
            raise

    def validate_response(self, request, response):
        self._current()
        if request is not self.pending or not isinstance(response, MctsResponse):
            raise ValueError("response does not belong to the current pending request")
        if (response.request_id != request.id or response.evaluator_identity != request.evaluator_identity
                or tuple(response.action_ids) != request.action_ids
                or self.bindings[request.binding].identity != request.evaluator_identity):
            raise ValueError("request, evaluator or action mapping identity mismatch")
        if (not isinstance(response.values, np.ndarray) or response.values.dtype != np.float64
                or response.values.ndim != 1 or not np.isfinite(response.values).all()):
            raise ValueError("response values must be a finite float64 vector")
        player = response.scalar_player
        if player is not None:
            player = _integer(player, "scalar player")
        self._native.respond(*request.id, request.kind, response.values, player, True)

    def resume(self, request, response):
        """Invalid responses preserve the pending request for retry."""
        self.validate_response(request, response)
        try:
            return self._consume(self._native.respond(*request.id, request.kind,
                                                     response.values, response.scalar_player))
        except Exception as error:
            self.error = str(error)
            self.close()
            raise

    def close(self):
        if not self._closed:
            self._closed = True
            self.pending = None
            self._native.cancel()

    cancel = close

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass(frozen=True)
class MctsGroup:
    """Compatible requests from different trees, with explicit search indices."""
    members: tuple[tuple[int, MctsRequest], ...]
    evaluator: object

    @property
    def rows(self):
        return sum(request.rows for _, request in self.members)


class MctsBatch:
    """Coordinate bounded independent searches; one pending evaluation per tree.

    Rust workers are visited sequentially. Python groups observations for inference,
    without promising parallel search throughput or GPU execution.
    """
    def __init__(self, searches, *, max_searches=64, max_group_rows=4096,
                 max_bytes=64 * 1024 * 1024):
        self.searches = tuple(searches)
        if (not self.searches or len(self.searches) > _integer(max_searches, "max_searches")
                or len({id(s) for s in self.searches}) != len(self.searches)
                or any(not isinstance(s, MctsSearch) for s in self.searches)):
            raise ValueError("provide distinct searches within the coordinator limit")
        self.max_group_rows = _integer(max_group_rows, "max_group_rows")
        self.max_bytes = _integer(max_bytes, "max_bytes")
        if self.max_group_rows == 0 or self.max_bytes == 0:
            raise ValueError("group row and byte limits must be positive")
        self.errors = {}
        self.groups = ()

    def advance(self):
        requests = []
        for index, search in enumerate(self.searches):
            if index in self.errors or search.report is not None:
                continue
            try:
                search.advance()
                if search.pending is not None:
                    requests.append((index, search.pending))
            except Exception as error:
                self.errors[index] = str(error)
        groups = []
        by_key = {}
        for index, request in requests:
            search = self.searches[index]
            evaluator = search.bindings[request.binding]
            identity = request.observations[0].identity
            layout = tuple((name, array.dtype.str, array.shape)
                           for name, array in request.observations[0].payloads.items())
            key = (request.evaluator_identity, identity, layout, search.value_observers,
                   search.settings["value_completion"])
            if request.rows > self.max_group_rows or request.nbytes > self.max_bytes:
                raise MemoryError("one MCTS request exceeds the group limits")
            existing = by_key.get(key)
            if existing is None or (sum(r.rows for _, r in groups[existing][0]) + request.rows > self.max_group_rows
                                    or sum(r.nbytes for _, r in groups[existing][0]) + request.nbytes > self.max_bytes):
                by_key[key] = len(groups)
                groups.append(([(index, request)], evaluator))
            else:
                groups[existing][0].append((index, request))
        self.groups = tuple(MctsGroup(tuple(members), evaluator) for members, evaluator in groups)
        return self.groups

    def resume(self, group, responses):
        responses = tuple(responses)
        if not any(group is current for current in self.groups) or len(responses) != len(group.members):
            raise ValueError("foreign group or incorrect response count")
        # Prevalidate every member before committing any evaluation in this group.
        for (index, request), response in zip(group.members, responses):
            self.searches[index].validate_response(request, response)
        for (index, request), response in zip(group.members, responses):
            try:
                self.searches[index].resume(request, response)
            except Exception as error:
                self.errors[index] = str(error)
        self.groups = tuple(current for current in self.groups if current is not group)

    def run(self):
        while True:
            groups = self.advance()
            if not groups and all(search.report is not None or index in self.errors
                                  for index, search in enumerate(self.searches)):
                return tuple(search.report for search in self.searches)
            for group in groups:
                requests = tuple(request for _, request in group.members)
                responses = group.evaluator.evaluate(requests, max_bytes=self.max_bytes)
                self.resume(group, responses)

    def close(self):
        for search in self.searches:
            search.close()
        self.groups = ()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
