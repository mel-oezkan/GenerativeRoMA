"""R1: reconstruction probes on frozen features.

  probe  the RAE stage-1 probe (decoder, training loop, eval, precompute)

The same probe measures stock RoMaV2 taps and the R2/R3 checkpoints — only
the feature cache it reads changes (cfg.cache_dir).
"""
