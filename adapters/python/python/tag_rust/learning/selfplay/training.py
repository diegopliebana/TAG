"""Completed-cycle policy iteration using Rust search and CPU PyTorch updates."""
from dataclasses import replace
import json
import time
import torch
from tag_rust import _engine
from tag_rust.engine.api import create
from tag_rust.learning.inputs import ModelContract,ValueSpec
from tag_rust.learning.network import PolicyValueNetwork
from tag_rust.learning.snapshot import FrozenModel
from tag_rust.learning.artifacts import model_document,load_document
from tag_rust.learning.training import runtime_identity
from tag_rust.search.mcts import MctsSearch
from .settings import SelfPlaySettings
from .seeds import workflow_seed,seed_version,game_parameters,generator
from .data import ReplayBuffer
from .collection import collect_games,start_search,neural_evaluator
from .loss import loss_terms
from .evaluation import evaluate
from ..outcomes import seat_outcome_counts


class SelfPlaySession:
    """One game/configuration, bounded examples and independently owned model state.

    A collection or training failure blocks checkpoints and further cycles. The
    last completed on-disk checkpoint remains the recovery point. Evaluation
    failures are diagnostics and do not undo successfully completed updates.
    """
    def __init__(self,settings=None,*,initial_model=None):
        settings = SelfPlaySettings() if settings is None else settings
        if isinstance(settings,dict):
            settings = SelfPlaySettings.from_dict(settings)
        if not isinstance(settings,SelfPlaySettings):
            raise TypeError("settings must be SelfPlaySettings or a configuration object")
        # Copy mutable nested dictionaries and resolve the automatic seed once.
        settings = SelfPlaySettings.from_dict(settings.document())
        if settings.seed is None:
            settings = replace(settings,seed=_engine.automatic_search_seed())
        self.settings = settings
        self._configuration = json.dumps(settings.document(),sort_keys=True,allow_nan=False)
        torch.set_num_threads(settings.limits.threads)
        torch.use_deterministic_algorithms(True)
        self.runtime = runtime_identity()
        with create(settings.game,game_parameters(settings,0),players=settings.players,max_cells=settings.limits.max_cells,
                    max_bytes=settings.limits.max_request_bytes) as game:
            if settings.evaluation.enabled and settings.players != 2 and settings.game not in ("cantstop", "loveletter"):
                raise ValueError("candidate/opponent evaluation currently requires two players")
            players = None if settings.model.value_output == "observer" else (settings.model.value_players or tuple(range(settings.players)))
            if players is not None and set(players) != set(range(settings.players)):
                raise ValueError("value_players must cover every game player exactly once")
            value = ValueSpec(players=players,gamma=1.,discount_clock="environment-action")
            contract = ModelContract.from_spec(game.spec(),payloads=settings.model.payloads,value=value)
            if contract.identity.player_count != settings.players:
                raise ValueError("game encoding and configured player counts differ")
            if contract.actions.mapping == "candidates":
                raise ValueError("self-play requires an implemented fixed-action policy encoder")
            widths = (contract.input_size,*settings.model.hidden_sizes)
            parameter_count = sum((left+1)*right for left,right in zip(widths,widths[1:]))
            parameter_count += (widths[-1]+1)*(contract.actions.size+contract.value.size)
            if parameter_count > settings.limits.max_model_parameters:
                raise MemoryError("model exceeds max_model_parameters")
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(workflow_seed(settings,"initialization",purpose="model"))
                if initial_model is None:
                    self.model = PolicyValueNetwork(contract,settings.model.hidden_sizes)
                else:
                    if type(initial_model) is not PolicyValueNetwork:
                        raise TypeError("initial_model must be the portable PolicyValueNetwork")
                    if tuple(initial_model.hidden_sizes) != settings.model.hidden_sizes:
                        raise ValueError("initial model architecture differs from settings")
                    self.model = load_document(model_document(initial_model),expected_contract=contract)
                self.torch_rng = torch.get_rng_state().clone()
            self.initial_identity = FrozenModel(self.model).identity
            prototype = dict(settings.search)
            prototype["seed"] = 0
            # Validate evaluator construction without performing a search or
            # advancing any logical self-play RNG stream.
            with start_search(game,settings,prototype,neural_evaluator(FrozenModel(self.model))):
                pass
            if settings.evaluation.enabled:
                if settings.game in ("cantstop", "loveletter"):
                    from .lineup_evaluation import validate_lineup
                    validate_lineup(settings,FrozenModel(self.model),game)
                else:
                    opponent = dict(settings.evaluation.opponent_search)
                    opponent["seed"] = 0
                    with MctsSearch(game,opponent):
                        pass
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=settings.optimization.learning_rate)
        self.buffer = ReplayBuffer(settings.data,contract)
        self.sample_rng = generator(workflow_seed(settings,"training",purpose="minibatches"))
        self.completed_cycles = 0
        self.next_game_id = 0
        self.updates = 0
        self._history = []
        self.failed = False
        self.busy = False

    def check_boundary(self):
        if self.busy or self.failed:
            raise RuntimeError("self-play checkpoint requires a successful completed-cycle boundary")
        if runtime_identity() != self.runtime:
            raise ValueError("self-play CPU/runtime settings changed during the session")
        if json.dumps(self.settings.document(),sort_keys=True,allow_nan=False) != self._configuration:
            raise ValueError("self-play settings changed during the session")
        if (self.completed_cycles != len(self._history) or self.next_game_id != self.completed_cycles*self.settings.games_per_cycle
            or self.updates != self.completed_cycles*self.settings.optimization.updates_per_cycle):
            raise ValueError("self-play counters differ from completed cycles")
        expected = self._history[-1]["model_identity"] if self._history else self.initial_identity
        if FrozenModel(self.model).identity != expected:
            raise ValueError("model changed outside a completed self-play cycle")
        if self._history:
            if self.buffer.digest() != self._history[-1]["buffer_digest"]:
                raise ValueError("training buffer changed outside a completed cycle")
        elif len(self.buffer):
            raise ValueError("initial session unexpectedly contains training positions")

    @property
    def history(self):
        return json.loads(json.dumps(self._history,allow_nan=False))

    def _optimize(self):
        settings = self.settings
        records = []
        started = time.perf_counter()
        self.model.train()
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.torch_rng)
            for _ in range(settings.optimization.updates_per_cycle):
                positions,indices = self.buffer.sample(settings.optimization.batch_size,self.sample_rng)
                self.optimizer.zero_grad(set_to_none=True)
                policy,value = loss_terms(self.model,positions,max_bytes=settings.limits.max_request_bytes)
                loss = settings.optimization.policy_weight*policy+settings.optimization.value_weight*value
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite self-play objective")
                loss.backward()
                if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in self.model.parameters()):
                    raise ValueError("nonfinite self-play gradient")
                self.optimizer.step()
                if any(not torch.isfinite(tensor).all() for tensor in self.model.state_dict().values()):
                    raise ValueError("nonfinite model after optimizer update")
                for state in self.optimizer.state.values():
                    if any(isinstance(tensor,torch.Tensor) and not torch.isfinite(tensor).all() for tensor in state.values()):
                        raise ValueError("nonfinite optimizer state")
                self.updates += 1
                records.append({"update":self.updates,"policy_loss":float(policy.detach()),
                    "value_loss":float(value.detach()),"loss":float(loss.detach()),"sample_indices":list(indices)})
            self.torch_rng = torch.get_rng_state().clone()
        return records,(time.perf_counter()-started)*1000

    def run_cycle(self):
        self.check_boundary()
        if self.completed_cycles >= self.settings.cycles:
            raise ValueError("all configured self-play cycles are complete")
        self.busy = True
        started = time.perf_counter()
        try:
            snapshot = FrozenModel(self.model)
            collected = collect_games(self.settings,snapshot,first_game_id=self.next_game_id)
            self.buffer.add_group(collected.positions)
            updates,training_ms = self._optimize()
            trained = FrozenModel(self.model)
            self.completed_cycles += 1
            first_game = self.next_game_id
            self.next_game_id += self.settings.games_per_cycle
            try:
                evaluation = evaluate(self.settings,trained,self.completed_cycles-1)
            except Exception as error:
                evaluation = {"enabled":self.settings.evaluation.enabled,"completed_games":0,
                              "failures":[{"error":str(error)}]}
            report = {"cycle":self.completed_cycles,"before_identity":snapshot.identity,"model_identity":trained.identity,
                "first_game_id":first_game,"next_game_id":self.next_game_id,"updates":self.updates,
                "games":[{"game_id":game.game_id,"outcomes":list(game.outcomes),"moves":len(game.positions),
                          **({"first_player":game.positions[0].actor,
                              "actions":[position.action for position in game.positions]}
                             if self.settings.game == "loveletter" else {}),
                          **({"game_seed":game.game_seed} if game.game_seed is not None else {})} for game in collected.games],
                "collection":collected.metrics,"optimization":updates,"training_ms":training_ms,
                "buffer_positions":len(self.buffer),"buffer_bytes":self.buffer.nbytes,"dropped_positions":self.buffer.dropped,
                "buffer_digest":self.buffer.digest(),"evaluation":evaluation,"elapsed_ms":(time.perf_counter()-started)*1000}
            if seed_version(self.settings) == 2:
                report.update(seat_outcome_counts((game.outcomes for game in collected.games),self.settings.players))
            self._history.append(report)
        except BaseException:
            self.failed = True
            raise
        finally:
            self.busy = False
        return json.loads(json.dumps(report,allow_nan=False))

    def run(self,*,max_cycles=None):
        if max_cycles is not None and (type(max_cycles) is not int or max_cycles < 0):
            raise ValueError("max_cycles must be nonnegative or null")
        remaining = self.settings.cycles-self.completed_cycles
        for _ in range(remaining if max_cycles is None else min(remaining,max_cycles)):
            yield self.run_cycle()

    def provenance(self):
        self.check_boundary()
        result = {"algorithm":"mcts-selfplay","version":1,"seed_derivation_version":seed_version(self.settings),
                "master_seed":self.settings.seed,"completed_cycles":self.completed_cycles,
                "next_game_id":self.next_game_id,"updates":self.updates,
                "initial_identity":self.initial_identity,"settings":self.settings.document(),
                "model_context":{"observers":list(range(self.settings.players)),
                    "turn_context":"any_turn" if self.model.contract.value.players is None else "own_turn"}}
        if self.settings.game == "loveletter":
            result["information_access"] = "privileged-research"
        return result
