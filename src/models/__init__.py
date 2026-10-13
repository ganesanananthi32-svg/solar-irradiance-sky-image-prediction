"""Models module: Standard baselines, Gao & Liu (2022) baseline, and novel 2026+ Diff-CSM."""
from .baselines import PersistenceModel, SmartPersistenceModel, ClearSkyOnlyModel, DirectRegressionBaseline
from .residual_network import GaoLiuResidualNetwork
from .novel_sun_attention_net import DualStreamCircumsolarResidualNet
from .losses import PhysicsRampLoss, QuantilePinballLoss
from .diff_csm_2026 import DiffCSMPolarAttentionNowcaster, SunCentricPolarWarper, DiffClearSkyModel

__all__ = [
    "PersistenceModel",
    "SmartPersistenceModel",
    "ClearSkyOnlyModel",
    "DirectRegressionBaseline",
    "GaoLiuResidualNetwork",
    "DualStreamCircumsolarResidualNet",
    "PhysicsRampLoss",
    "QuantilePinballLoss",
    "DiffCSMPolarAttentionNowcaster",
    "SunCentricPolarWarper",
    "DiffClearSkyModel",
]
