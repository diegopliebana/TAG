"""Search-policy cross-entropy and per-player terminal-outcome regression."""
import numpy as np
import torch
from tag_rust.learning.inputs import pack_observations
from .data import validate_example


def loss_terms(model,positions,*,max_bytes=64*1024*1024):
    positions = tuple(positions)
    if not positions:
        raise ValueError("provide training positions")
    observations,policy_rows,policies,values = [],[],[],[]
    contract = model.contract
    for position in positions:
        validate_example(position,contract,max_bytes=max_bytes)
        for observation in position.observations:
            if observation.observer == position.actor:
                policy_rows.append(len(observations))
                policies.append(position.policy)
            values.append(position.terminal_values[observation.observer] if contract.value.players is None
                          else position.terminal_values[list(contract.value.players)])
            observations.append(observation)
    batch = pack_observations(contract,observations,max_bytes=max_bytes)
    logits,predicted_values = model({name:torch.from_numpy(array.copy()) for name,array in batch.payloads.items()},
                                   torch.from_numpy(batch.context.copy()))
    expected_shape = (len(observations),) if contract.value.players is None else (len(observations),contract.value.size)
    if (logits.shape != batch.legal_mask.shape or predicted_values.shape != expected_shape
        or not torch.isfinite(logits).all() or not torch.isfinite(predicted_values).all()):
        raise ValueError("invalid training policy/value outputs")
    rows = torch.tensor(policy_rows,dtype=torch.int64)
    mask = torch.from_numpy(batch.legal_mask[policy_rows].copy())
    log_probabilities = torch.log_softmax(logits[rows].masked_fill(~mask,-torch.inf),dim=1)
    # Illegal actions have no target mass. Replacing their log probabilities
    # avoids the undefined product 0 * -infinity in the cross-entropy sum.
    log_probabilities = log_probabilities.masked_fill(~mask,0.)
    targets = torch.from_numpy(np.stack(policies).copy())
    policy_loss = -(targets*log_probabilities).sum(dim=1).mean()
    value_targets = torch.from_numpy(np.asarray(values,dtype=np.float32))
    value_loss = ((predicted_values-value_targets)**2).mean()
    if not torch.isfinite(policy_loss) or not torch.isfinite(value_loss):
        raise ValueError("nonfinite self-play loss")
    return policy_loss,value_loss
