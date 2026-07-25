"""Train-log parsing for the loss-curve figures.

Two log formats, one parser: the R1 probe logs recon terms only, the R2/R3
training logs prepend the matching terms. The GAN tail (gan/lam/d) is
optional in both — it only appears once the corresponding phase starts.
"""

import re

# [mv] step 3600 l1 0.1853 lpips 0.6342 gan 0.9714 lam 0.212 d 0.2623
PROBE_RE = re.compile(
    r"\[\w+\] step (?P<step>\d+) l1 (?P<l1>[\d.]+) lpips (?P<lpips>[\d.]+)"
    r"(?: gan (?P<gan>[\d.]+) lam (?P<lam>[\d.]+))?(?: d (?P<d>[\d.]+))?"
)
PROBE_TERMS = {
    "l1": "L1",
    "lpips": "LPIPS (VGG)",
    "gan": "G adversarial",
    "lam": "adaptive weight λ",
    "d": "D hinge",
}

# [joint] step 200 attn 2.426 warp 0.1481 conf 0.1902 l1 0.1896 lpips 0.5859
TRAIN_RE = re.compile(
    r"\[\w+\] step (?P<step>\d+) attn (?P<attn>[\d.]+) warp (?P<warp>[\d.]+)"
    r" conf (?P<conf>[\d.]+) l1 (?P<l1>[\d.]+) lpips (?P<lpips>[\d.]+)"
    r"(?: gan (?P<gan>-?[\d.]+) lam (?P<lam>[\d.]+))?(?: d (?P<d>[\d.]+))?"
)
TRAIN_TERMS = {
    "attn": "attn CE (/16)",
    "warp": "warp Charbonnier (/4)",
    "conf": "conf BCE",
    "l1": "recon L1",
    "lpips": "recon LPIPS (VGG)",
    "gan": "GAN (vanilla G)",
    "lam": "adaptive lambda",
    "d": "disc hinge",
}


def parse_log(path, line_re, terms):
    """-> {term: (steps, values)}, resume duplicates collapsed (last wins).

    Terms that are identically zero are dropped: an arm that doesn't carry a
    loss logs it as a literal 0 (the match arm logs l1/lpips 0.0000), and a
    flat zero line in a panel reads as a result rather than an absence.
    """
    by_step = {}
    for line in path.read_text(errors="replace").splitlines():
        m = line_re.search(line)
        if m:
            by_step[int(m["step"])] = {
                k: float(m[k]) for k in terms if m[k] is not None
            }
    series = {}
    for step in sorted(by_step):
        for k, v in by_step[step].items():
            series.setdefault(k, ([], []))
            series[k][0].append(step)
            series[k][1].append(v)
    return {k: sv for k, sv in series.items() if any(v != 0 for v in sv[1])}


def parse_arm_logs(run_dir, line_re, terms):
    """{arm: series} from run_dir/train_<arm>.log (arm from the filename —
    mv_selfpair logs tag themselves [mv])."""
    arms = {}
    for log in sorted(run_dir.glob("train_*.log")):
        series = parse_log(log, line_re, terms)
        if series:
            arms[log.stem.removeprefix("train_")] = series
    return arms
