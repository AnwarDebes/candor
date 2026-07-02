"""CANDOR: Concept-ANchored Disentangled Output Routing.

A reference implementation of *interpretability by construction*: a network whose
output-relevant computation is routed through Legible Bottlenecks (sparse, typed,
persistently-named concept channels) and which emits, at every forward pass, a
checkable Faithfulness Certificate bounding its non-legible ("dark") computation.

See the paper in ``paper/candor.tex`` ("The Explanation Is the Forward Pass:
Interpretability by Construction with Certified Legible Bottlenecks").
"""
from .bottleneck import BottleneckOutput, LegibleBottleneck, topk_mask
from .certificate import Certificate, certify
from .data import (ModAddData, PlantedData, SeqData, modular_addition,
                   planted_concepts, sequence_max, split, trojan_concepts)
from .losses import (LossBreakdown, LossWeights, candor_lm_loss, candor_loss,
                     leak_energy, lm_cross_entropy)
from .metrics import (causal_faithfulness, concept_activations, concept_recovery,
                      stability, task_accuracy)
from .model import (LegibleLMTransformer, LegibleMLP, LegibleTransformer,
                    OpaqueLMTransformer, OpaqueMLP)
from .train import (TrainConfig, device_auto, fit_sae, sae_codes, train_candor,
                    train_opaque)

__version__ = "1.3.0"

__all__ = [
    "LegibleBottleneck", "BottleneckOutput", "topk_mask",
    "LegibleMLP", "LegibleTransformer", "OpaqueMLP",
    "LegibleLMTransformer", "OpaqueLMTransformer",
    "candor_loss", "candor_lm_loss", "lm_cross_entropy",
    "LossWeights", "LossBreakdown", "leak_energy",
    "certify", "Certificate",
    "planted_concepts", "trojan_concepts", "modular_addition", "sequence_max", "split",
    "PlantedData", "ModAddData", "SeqData",
    "concept_recovery", "concept_activations", "causal_faithfulness",
    "task_accuracy", "stability",
    "train_candor", "train_opaque", "fit_sae", "sae_codes", "TrainConfig",
    "device_auto",
    "__version__",
]
