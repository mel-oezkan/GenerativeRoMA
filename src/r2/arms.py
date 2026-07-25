"""Arm semantics: which losses, which modules, which schedules are live.

One place decides what "match" / "joint" / "recon" mean, so the training
loop, the post-hoc evals and the checkpoint loaders cannot disagree about
whether an arm has a decoder or ran the matching head.
"""

from dataclasses import dataclass

ARMS = ["match", "joint", "recon"]


@dataclass(frozen=True)
class ArmFlags:
    arm: str
    lam_rec: float
    with_decoder: bool
    use_matching: bool
    desc_only: bool
    skip_head: bool
    gan: bool


def arm_flags(cfg):
    arm = cfg.arm
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; have {ARMS}")
    desc_only = bool(cfg.model.desc_baseline)
    with_decoder = arm != "match"
    if desc_only and cfg.model.init == "pretrained":
        raise ValueError(
            "desc baseline has no mv_vit to load pretrained weights into"
        )
    return ArmFlags(
        arm=arm,
        # the match arm carries no recon term regardless of the configured weight
        lam_rec=0.0 if arm == "match" else float(cfg.loss.lam_rec),
        with_decoder=with_decoder,
        use_matching=arm != "recon",
        desc_only=desc_only,
        # pure RAE control: proj -> decoder only, matcher head never runs
        skip_head=desc_only and arm == "recon",
        gan=bool(with_decoder and arm in cfg.gan.arms),
    )
