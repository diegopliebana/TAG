"""Root-only exploration applied after compatible frozen-model inference."""
from dataclasses import replace
import json
import numpy as np
from .seeds import generator


class RootNoiseEvaluator:
    """Share inference across trees while retaining each decision's own noise stream.

    bind() supplies a logical decision's recorded seed. The evaluator identity
    declares post-processing settings; that per-decision seed belongs to behavior
    provenance, independently of the frozen model identity.
    """
    def __init__(self,base,settings):
        self.base,self.settings = base,settings
        self.identity = json.dumps({"type":"root-dirichlet","version":1,
            "base":base.identity,"fraction":settings.noise_fraction,
            "alpha":settings.noise_alpha},sort_keys=True,separators=(",",":"))
        self._contexts = {}

    @property
    def payloads(self):
        return self.base.payloads

    def validate(self,*args):
        return self.base.validate(*args)

    def bind(self,request,seed):
        context = self._contexts.setdefault(request.id[0],{"seed":seed,"noise":None})
        if context["seed"] != seed:
            raise ValueError("root-noise decision seed changed")

    def release(self,search_ids):
        for search_id in search_ids:
            self._contexts.pop(search_id,None)

    def evaluate(self,requests,*,max_bytes):
        requests = tuple(requests)
        if any(request.evaluator_identity != self.identity for request in requests):
            raise ValueError("root-noise evaluator identity mismatch")
        ordinary = tuple(replace(request,evaluator_identity=self.base.identity) for request in requests)
        predictions = self.base.evaluate(ordinary,max_bytes=max_bytes)
        replies = []
        for request,prediction in zip(requests,predictions):
            values = prediction.values.copy()
            if request.kind == "policy":
                if request.policy_node is None:
                    raise ValueError("root noise requires explicit policy-node metadata")
                if request.policy_node == 0:
                    context = self._contexts.get(request.id[0])
                    if context is None:
                        raise ValueError("root-noise request has no logical decision seed")
                    if context["noise"] is None:
                        draw = generator(context["seed"]).dirichlet(np.full(len(request.action_ids),self.settings.noise_alpha))
                        context["noise"] = dict(zip(request.action_ids,draw))
                    if set(context["noise"]) != set(request.action_ids):
                        raise ValueError("root action identities changed within a decision")
                    noise = np.array([context["noise"][action] for action in request.action_ids])
                    fraction = self.settings.noise_fraction
                    values = (1-fraction)*values/values.sum() + fraction*noise
            replies.append(replace(prediction,evaluator_identity=self.identity,values=values))
        return tuple(replies)
