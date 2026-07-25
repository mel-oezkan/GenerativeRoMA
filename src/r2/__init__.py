"""R2/R3: joint matching + reconstruction training on CO3D pairs.

Modules
  dataset  pair filtering (test-frame / pose-quality / covisibility) and the
           desc + image + GT-warp dataset
  model    the Matcher (+ optional recon decoder) and checkpoint loading
  losses   attention CE, warp Charbonnier, confidence BCE
  metrics  the shared eval loop (aggregate and per-category)
  viz      warp panels + recon grids written next to the metrics
  train    the training loop
"""
