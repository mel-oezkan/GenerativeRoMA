from .scannet1500 import ScanNet1500 as ScanNet1500
from .mega1500 import Mega1500 as Mega1500

# Vendored-copy deviation: these two pull in optional third-party packages
# (wxbs_benchmark, and satast's data deps) that are not installed in the
# `cv` env. Importing them eagerly made `from romav2.benchmarks.mega1500
# import Mega1500` fail, so they degrade to None instead.
try:
    from .wxbs import WxBSBenchmark as WxBSBenchmark
except ImportError:  # pragma: no cover - optional dependency
    WxBSBenchmark = None
try:
    from .satast import SatAst as SatAst
except ImportError:  # pragma: no cover - optional dependency
    SatAst = None
