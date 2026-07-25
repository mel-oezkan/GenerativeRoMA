"""Build docs/matcher_results.html: the standalone (base64-embedded)
matcher-results report.

Covers what the metrics mean (EPE / PCK@1 / PCK@5), the in-domain r3-4cat
results for every arm, the released-checkpoint references with and without its
refiners, and the held-out-category generalization sweep. Numbers are written
out by hand from the metrics JSONs so the page reads as a report; re-run an
eval, then update the tables here."""

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.paths import DOCS_DIR, FIGURES_DIR  # noqa: E402

OUT = DOCS_DIR / "matcher_results.html"

IMGS = {
    "pck": FIGURES_DIR / "r3/fig_epe_pck_example.png",
    "reconq": FIGURES_DIR / "r3/fig_recon_quality.png",
    "gen": FIGURES_DIR / "r3/fig_generalization.png",
}


def b64(p):
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


CSS = """
:root { color-scheme: light dark;
  --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --rule:#e1e0d9; --card:#ffffff; --accent:#2a78d6; --code:#f4f3ef; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16161a; --ink:#f2f1ee; --ink2:#c6c4bd; --muted:#8f8d86;
    --rule:#2e2e33; --card:#1e1e23; --accent:#6aa8ee; --code:#232329; } }
* { box-sizing: border-box; }
body { margin:0; padding:36px 20px 72px; background:var(--bg); color:var(--ink);
  font:16px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 940px; margin: 0 auto; }
h1 { font-size:1.72rem; line-height:1.25; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:1.16rem; margin:2.4em 0 0.5em; padding-top:0.7em;
  border-top:1px solid var(--rule); letter-spacing:-0.005em; }
h3 { font-size:1rem; margin:1.6em 0 0.4em; }
p, li { color:var(--ink2); }
.lede { font-size:1.03rem; color:var(--ink2); margin:0 0 4px; }
.meta { color:var(--muted); font-size:0.85rem; margin:0 0 8px; }
strong, b { color:var(--ink); }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
code { background:var(--code); padding:1px 5px; border-radius:4px; font-size:0.87em; }
pre { background:var(--code); padding:14px 16px; border-radius:8px; overflow-x:auto;
  font-size:0.83rem; line-height:1.5; border:1px solid var(--rule); }
pre code { background:none; padding:0; font-size:1em; }
figure { margin:22px 0 8px; }
figure img { width:100%; height:auto; display:block; border:1px solid var(--rule);
  border-radius:8px; background:#fcfcfb; }
figcaption { color:var(--muted); font-size:0.85rem; margin-top:8px; }
.scroll { overflow-x:auto; margin:16px 0; }
table { border-collapse:collapse; width:100%; min-width:640px; font-size:0.88rem; }
caption { caption-side:top; text-align:left; color:var(--muted);
  font-size:0.85rem; padding-bottom:8px; }
th { text-align:right; font-weight:600; color:var(--ink); padding:7px 10px;
  border-bottom:2px solid var(--rule); white-space:nowrap; }
td { text-align:right; padding:6px 10px; border-bottom:1px solid var(--rule);
  color:var(--ink2); font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { text-align:left; }
tbody tr:hover td { background:var(--code); }
td.na { color:var(--muted); }
.win { color:var(--ink); font-weight:600; }
.grp td { background:var(--code); color:var(--muted); font-size:0.8rem;
  text-transform:uppercase; letter-spacing:0.06em; }
.callout { background:var(--card); border:1px solid var(--rule);
  border-left:3px solid var(--accent); border-radius:6px; padding:14px 16px;
  margin:18px 0; font-size:0.94rem; }
.callout p { margin:0; }
ul { padding-left:1.15em; }
li { margin:0.3em 0; }
.foot { color:var(--muted); font-size:0.82rem; margin-top:2.6em;
  border-top:1px solid var(--rule); padding-top:14px; }
"""

BODY = f"""
<main>
<h1>Matcher results</h1>
<p class="meta">GenerativeRoMA · 320&nbsp;px, no refiners · in-domain r3-4cat
(209 pairs) · held-out categories co3d-unseen (413 pairs)</p>
<p class="lede">Every matcher we trained, on one page: what the metrics mean,
how the arms compare in domain, what the released checkpoint scores as a
reference, and how much of any of it survives a move to CO3D categories the
models never saw.</p>
<div class="callout"><p><b>Headline.</b> The reconstruction objective's effect
on matching is small and <i>domain-invariant</i> — whichever way joint-vs-match
falls in domain, it falls the same way on 17 unseen categories. The larger
effects are elsewhere: fine-tuning from the pretrained checkpoint beats
from-scratch training by ~4&times; out of domain, and the frozen-DINOv3
baseline <i>overtakes</i> the deep from-scratch encoder once the categories
change.</p></div>

<h2>1. What the metrics mean</h2>
<p>For every covisible pixel in view&nbsp;A the model predicts where it lands in
view&nbsp;B. <b>EPE</b> (end-point error) is the mean Euclidean distance between
that prediction and the ground-truth location. <b>PCK@&tau;</b> ("percentage of
correct keypoints") is the fraction of pixels whose error is under &tau;
pixels &mdash; PCK@1 uses &tau;&nbsp;=&nbsp;1&nbsp;px, PCK@5 uses
&tau;&nbsp;=&nbsp;5&nbsp;px.</p>
<pre><code># experiments/r3_eval_cats.py:73
epe4 = ((preds["warp_AB"][0] - batch["warp4_ab"][0]) / 2 * SIZE).norm(dim=-1)[mask]
"epe":  epe4.mean().item(),
"pck1": (epe4 &lt; 1).float().mean().item(),
"pck5": (epe4 &lt; 5).float().mean().item(),</code></pre>
<p>RoMaV2's warps live in a normalised <code>[-1, 1]</code> grid, so
<code>/ 2 * SIZE</code> converts to pixels at 320. Only pixels passing the
covisibility mask count, and pairs with fewer than 16 valid pixels are skipped
entirely &mdash; that is why n&nbsp;=&nbsp;209 of the 237 sampled pairs.</p>
<ul>
<li><b>EPE is an average, so a few catastrophic pixels dominate it.</b> A model
that nails 90&nbsp;% of pixels and sends 10&nbsp;% to the wrong side of the
object can post a worse EPE than a uniformly mediocre one.</li>
<li><b>PCK@1 is precision, PCK@5 is reliability.</b> &tau;&nbsp;=&nbsp;1&nbsp;px
asks for sub-pixel accuracy (what pose estimation and triangulation need);
&tau;&nbsp;=&nbsp;5&nbsp;px (~1.6&nbsp;% of the image) asks whether the match
found the right <i>place</i> at all.</li>
</ul>

<h2>2. A worked example</h2>
<p>Two toy models, 12 evaluated pixels each, errors in px at 320:</p>
<div class="scroll">
<table>
  <caption>Same 12 pixels, two error profiles &mdash; the metrics disagree.</caption>
  <thead><tr><th>model</th><th>per-pixel errors (px)</th><th>EPE &darr;</th>
    <th>PCK@1 &uarr;</th><th>PCK@5 &uarr;</th></tr></thead>
  <tbody>
    <tr><td>A &mdash; accurate, 2 blowouts</td>
      <td style="text-align:left">0.4 0.6 0.8 1.2 1.5 2.0 2.4 3.1 4.2 4.8 <b>30 45</b></td>
      <td>8.00</td><td class="win">0.25</td><td class="win">0.83</td></tr>
    <tr><td>B &mdash; uniformly mediocre</td>
      <td style="text-align:left">3.0 4.0 4.6 4.9 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0</td>
      <td class="win">6.21</td><td>0.00</td><td>0.33</td></tr>
  </tbody>
</table>
</div>
<figure>
  <img src="{{PCK}}" alt="Three panels: scatter of prediction errors for two toy
    models around the ground-truth match with tolerance rings at 1 and 5 px, and
    a PCK-versus-threshold curve showing the two models crossing near 8 px.">
  <figcaption>Left/middle: each dot is one evaluated pixel, placed at its error
    offset from the true correspondence (errors past the panel edge are clamped
    to the rim and labelled). Right: PCK@&tau; is literally the CDF of those
    errors; EPE is its mean. Where the curves cross is why the two metrics rank
    the models differently.</figcaption>
</figure>
<div class="callout"><p>By EPE, model&nbsp;B wins (6.21 &lt; 8.00). By PCK@5,
model&nbsp;A wins by 2.5&times;. Both readings are correct. A finds the right
location for 10 of 12 pixels and fails completely on 2 &mdash; those two
outliers contribute 78&nbsp;% of its total error. B never blows up but never
really succeeds either. Report only EPE and you pick B, whose warp field is
useless downstream.</p></div>

<h2>3. In-domain results (r3-4cat)</h2>
<p>All arms below are trained on <code>r3-4cat</code> at an equal 12k-step
budget. "deep" = the full RoMaV2 <code>mv_vit</code> encoder; "DINOv3" = the
frozen descriptors through a linear 2048&rarr;1024 projection in place of
<code>mv_vit</code>. Near/far split the eval pairs by baseline.</p>
<div class="scroll">
<table>
  <caption>Matching accuracy, 209 held-out pairs (coarse EPE = the stride-16
    global stage, before fine refinement).</caption>
  <thead><tr><th>arm</th><th>EPE all</th><th>EPE near</th><th>EPE far</th>
    <th>PCK@1 all</th><th>PCK@5 all</th><th>PCK@5 near</th><th>PCK@5 far</th>
    <th>coarse EPE far</th></tr></thead>
  <tbody>
    <tr class="grp"><td colspan="9">deep encoder (mv_vit, from scratch)</td></tr>
    <tr><td>match</td><td>7.66</td><td>2.02</td><td>14.30</td><td>0.130</td>
      <td>0.678</td><td>0.959</td><td>0.347</td><td>16.68</td></tr>
    <tr><td>joint 1-view (+recon+GAN)</td><td class="win">7.02</td><td>2.02</td>
      <td class="win">12.92</td><td>0.132</td><td class="win">0.686</td>
      <td class="win">0.961</td><td class="win">0.361</td><td>15.29</td></tr>
    <tr><td>joint 2-view (v2)</td><td>7.83</td><td>2.04</td><td>14.64</td>
      <td>0.132</td><td>0.677</td><td>0.956</td><td>0.348</td><td>16.62</td></tr>
    <tr class="grp"><td colspan="9">frozen DINOv3 desc + linear</td></tr>
    <tr><td>match</td><td>10.34</td><td>2.04</td><td>20.12</td><td>0.134</td>
      <td>0.582</td><td>0.955</td><td>0.143</td><td>60.15</td></tr>
    <tr><td>joint (+recon+GAN)</td><td>10.50</td><td>2.10</td><td>20.39</td>
      <td>0.122</td><td>0.580</td><td>0.953</td><td>0.141</td><td>60.65</td></tr>
    <tr><td>recon only</td><td class="na" colspan="8">no matching pipeline</td></tr>
    <tr class="grp"><td colspan="9">released romav2.pt, zero-shot reference</td></tr>
    <tr><td>pretrained, no refiners</td><td>4.46</td><td>2.15</td><td>7.17</td>
      <td>0.109</td><td>0.802</td><td>0.976</td><td>0.598</td><td>9.16</td></tr>
    <tr><td>pretrained + refiners (full system)</td><td class="win">3.24</td>
      <td class="win">0.90</td><td class="win">5.99</td><td class="win">0.506</td>
      <td class="win">0.874</td><td class="win">0.998</td><td class="win">0.729</td>
      <td>9.16</td></tr>
  </tbody>
</table>
</div>

<h3>The reference rows, read carefully</h3>
<p>The released checkpoint is a <b>reference, not an equal-budget arm</b>: it was
trained on far more data at higher resolution, and it is zero-shot on CO3D here.
Our arms have no refiners, so <b>"pretrained, no refiners" is the like-for-like
row</b> &mdash; the same coarse-only stride-4 warp pipeline our matchers run
(<code>--arm pretrained</code>). <b>"+ refiners"</b> adds the released
ConvRefiner stack (patch 4&nbsp;&rarr;&nbsp;2&nbsp;&rarr;&nbsp;1, turbo setting
at 320&nbsp;px, <code>--arm pretrained-refined</code>); its 320&nbsp;px output
warp is resampled to the same stride-4 grid the GT lives on.</p>
<div class="callout"><p><b>The refiners are almost entirely a sub-pixel
mechanism.</b> They cut EPE 4.46&nbsp;&rarr;&nbsp;3.24 and lift PCK@5 only
0.802&nbsp;&rarr;&nbsp;0.874 &mdash; but PCK@1 goes
<b>0.109&nbsp;&rarr;&nbsp;0.506</b>, a 4.6&times; jump. They do not find matches
the coarse stage missed; they sharpen matches it already had. The coarse EPE
column is bit-identical (9.16 far) because refiners never touch that stage.
This is the cleanest illustration on this page of why one metric is not
enough: judged on PCK@5 the refiners look like a 7-point detail, judged on
PCK@1 they are the single largest effect in the table.</p></div>
<p>Two numerical caveats on this pair, both measured rather than assumed. The
refined run recomputes descriptors in fp32, while the <code>pretrained</code>
row reads the fp16 <code>r2_desc_320</code> cache; running both fp32 in one
process gives a pre-refiner control of 4.50&nbsp;/&nbsp;2.15&nbsp;/&nbsp;7.26,
so the cache costs ~0.04&nbsp;px and the refiner delta above is if anything
slightly understated. Separately, the released config runs
<code>mv_vit</code>/DPT under bf16 autocast; that is disabled here to match
every other eval in the repo (Pascal has no native bf16), and leaving it on
costs ~0.13&nbsp;px on far pairs.</p>
<div class="scroll">
<table>
  <caption>Held-out reconstruction of the train-time decoders, 237 pairs, view A
    (the decoder's train target in every arm).</caption>
  <thead><tr><th>arm</th><th>PSNR &uarr;</th><th>SSIM &uarr;</th><th>LPIPS &darr;</th></tr></thead>
  <tbody>
    <tr><td>deep joint 1-view</td><td>21.41</td><td>0.531</td><td>0.198</td></tr>
    <tr><td>deep joint 2-view (v2)</td><td>21.71</td><td>0.542</td><td>0.181</td></tr>
    <tr><td>DINOv3 joint</td><td>23.06</td><td>0.597</td><td>0.137</td></tr>
    <tr><td>DINOv3 recon only (pure RAE)</td><td class="win">23.73</td>
      <td class="win">0.620</td><td class="win">0.122</td></tr>
    <tr><td>pretrained romav2.pt</td>
      <td class="na" colspan="3">no train-time decoder &mdash; its appearance
        content is measured post-hoc instead: 13.29&nbsp;dB / 0.141 /
        0.631 by the R1 mv-token probe (different protocol, hydrant-full)</td></tr>
  </tbody>
</table>
</div>
<figure>
  <img src="{{RECONQ}}" alt="Three bar panels comparing six arms on recon PSNR,
    recon LPIPS and far-pair EPE; match-only arms have no decoder and the
    recon-only arm has no matcher.">
  <figcaption>docs/figures/r3/fig_recon_quality.png &mdash; colour encodes the
    arm (blue matching-only, green joint, violet recon-only), x position groups
    the encoder family. Panels an arm cannot have are marked rather than
    dropped. The two dashed lines in the EPE panel are the zero-shot released
    checkpoint with and without its refiners; they are references, not
    equal-budget arms.</figcaption>
</figure>

<h2>4. Generalization: 17 held-out categories</h2>
<p>Everything above is measured on the same four categories the models trained
on (held-out <i>sequences</i>, but hydrant/bench/toybus/toytruck throughout), so
it cannot separate "the recon objective preserves appearance" from "the recon
objective fits CO3D turntable statistics". The <code>co3d-unseen</code> split
fixes the weaker half of that: 17 CO3D categories disjoint from every training
category (apple, ball, book, bowl, broccoli, cake, donut, mouse, orange, plant,
remote, skateboard, suitcase, teddybear, toaster, toytrain, vase), 413 pairs,
identical protocol and metrics. <b>Never trained on.</b></p>
<div class="scroll">
<table>
  <caption>Category transfer, 413 pairs. "macro EPE" averages the per-category
    means so the three categories with extra sequences cannot dominate.</caption>
  <thead><tr><th>arm</th><th>EPE all</th><th>EPE near</th><th>EPE far</th>
    <th>macro EPE</th><th>PCK@1</th><th>PCK@5</th>
    <th>in-domain EPE</th></tr></thead>
  <tbody>
    <tr class="grp"><td colspan="8">deep encoder, from scratch (12k)</td></tr>
    <tr><td>v1 match</td><td>18.56</td><td>2.69</td><td>35.29</td><td>17.80</td>
      <td>0.095</td><td>0.529</td><td>7.66</td></tr>
    <tr><td>v1 joint (+recon+GAN)</td><td>17.95</td><td>2.63</td><td>34.10</td>
      <td>17.02</td><td>0.092</td><td>0.530</td><td>7.02</td></tr>
    <tr><td>v2 joint (2-view)</td><td>17.59</td><td>2.70</td><td>33.30</td>
      <td>16.97</td><td>0.094</td><td>0.522</td><td>7.83</td></tr>
    <tr class="grp"><td colspan="8">frozen DINOv3 desc + linear (12k)</td></tr>
    <tr><td>desc match</td><td class="win">15.81</td><td>2.33</td>
      <td class="win">30.02</td><td class="win">15.08</td><td>0.102</td>
      <td>0.517</td><td>10.34</td></tr>
    <tr><td>desc joint</td><td>15.85</td><td>2.37</td><td>30.07</td><td>15.14</td>
      <td>0.098</td><td>0.515</td><td>10.50</td></tr>
    <tr class="grp"><td colspan="8">fine-tuned from romav2.pt (6k)</td></tr>
    <tr><td>ft match</td><td class="win">4.47</td><td class="win">0.86</td>
      <td class="win">8.28</td><td class="win">4.25</td><td>0.397</td>
      <td class="win">0.803</td><td>2.64</td></tr>
    <tr><td>ft joint (+recon)</td><td>4.66</td><td>0.87</td><td>8.66</td>
      <td>4.41</td><td>0.387</td><td>0.794</td><td>2.74</td></tr>
    <tr class="grp"><td colspan="8">released romav2.pt, zero-shot</td></tr>
    <tr><td>pretrained, no refiners</td><td>7.76</td><td>2.12</td><td>13.70</td>
      <td>7.36</td><td>0.105</td><td>0.731</td><td>4.46</td></tr>
    <tr><td>pretrained + refiners</td><td>6.74</td><td>0.91</td><td>12.89</td>
      <td>6.34</td><td class="win">0.439</td><td>0.789</td><td>3.24</td></tr>
  </tbody>
</table>
</div>
<figure>
  <img src="{{GEN}}" alt="Two bar panels: left compares each arm's in-domain and
    unseen-category EPE side by side; right shows the category-averaged
    unseen-category EPE per arm.">
  <figcaption>docs/figures/r3/fig_generalization.png &mdash; left: in domain
    (solid) vs held-out categories (faded), same protocol. Right: the same
    held-out result averaged per category, which is the sturdier ranking
    because the split is unbalanced (apple, suitcase and toytrain carry 5&ndash;6
    sequences from co3d_full; the other 14 categories 2 each from co3d_data).
    The two views agree on every ordering.</figcaption>
</figure>
<h3>What transfers</h3>
<ul>
<li><b>The joint-vs-match ordering is domain-invariant &mdash; the main
result.</b> From scratch, joint wins both in domain (7.02 vs 7.66) and out
(17.95 vs 18.56). Fine-tuned, match wins both in domain (2.64 vs 2.74) and out
(4.47 vs 4.66). Frozen-desc is a tie both times. So whatever the reconstruction
objective does to matching, it is not a property of the four training
categories.</li>
<li><b>The frozen-DINOv3 baseline and the deep encoder swap places.</b> In
domain the deep encoder wins by a wide margin (7.66 vs 10.34); on unseen
categories the frozen linear wins (15.81 vs 18.56, macro 15.1 vs 17.8). Part of
the deep <code>mv_vit</code>'s in-domain advantage was fitting those four
categories &mdash; and a frozen encoder cannot overfit them. This complicates
the "the collapse requires depth" reading from &sect;3: the depth that destroys
appearance is also the part that does not transfer.</li>
<li><b>Fine-tuning generalizes; from-scratch training does not.</b> ft-match
scores 4.47 on categories it never saw, better than the checkpoint it started
from (7.76) &mdash; four categories of in-domain fine-tuning improved matching
on 17 unrelated ones, so it learned something about CO3D capture rather than
about hydrants. Every from-scratch arm (16&ndash;19) is far <i>worse</i> than
zero-shot pretrained: 12k steps on four categories does not buy a general
matcher, whatever it does to the trade-off we are studying.</li>
<li><b>v2's two-view decoder is the best deep from-scratch arm out of
domain</b> (17.59 / macro 16.97) despite being the worst in domain (7.83). Weak
evidence that decoding both views regularizes rather than costs &mdash; worth
remembering next to its in-domain deficit.</li>
<li><b>Per-category spread is large and shared.</b> Every arm finds
<code>remote</code> and <code>skateboard</code> easiest and <code>bowl</code>,
<code>vase</code>, <code>apple</code> and <code>cake</code> hardest (ft-match:
2.0 on skateboard, 10.2 on bowl; v1-match: 3.6 on remote, 31.6 on cake) &mdash;
smooth, texture-poor, rotationally symmetric objects. The difficulty is a
property of the object, not of the arm.</li>
</ul>
<p>Still open: this is a category shift, not a regime shift &mdash; every pair
here is still an object-centric CO3D turntable. MegaDepth-1500 two-view pose
AUC (scene-level, wide baseline) is running; early smoke numbers suggest our
from-scratch arms land near the floor there (AUC@5 &asymp; 0.01 vs
&asymp; 0.45 for the released model with refiners), in which case that
benchmark will show a floor effect rather than a ranking.</p>

<h2>5. What the three metrics jointly say</h2>
<ul>
<li><b>Near pairs are a tie.</b> DINOv3 + linear matches short-baseline pairs as
well as 12 trained <code>mv_vit</code> blocks: EPE 2.04 vs 2.02, PCK@5 0.955 vs
0.959. Short-baseline matching needs no deep encoder.</li>
<li><b>Far pairs are where depth earns its keep, and EPE understates it.</b> Far
EPE 20.12 vs 14.30 is a 1.4&times; gap; far PCK@5 0.143 vs 0.347 is 2.4&times;.
Both models have heavy tails on wide baselines, and the tail is what EPE mostly
measures &mdash; PCK@5 shows that only 14&nbsp;% of DINOv3's far-pair pixels
land in the right neighbourhood at all.</li>
<li><b>The failure is in the coarse stage.</b> Far coarse EPE 60.15 vs
16.68&nbsp;px: global correspondence is essentially not found, and the fine warp
is refining from a wrong initialisation.</li>
<li><b>PCK@1 separates "matched" from "matched precisely".</b> Every coarse-only
arm sits at ~0.13 (and &lt;&nbsp;0.03 on far pairs) &mdash; including the
pretrained matcher at 0.109, which is otherwise far the strongest of them. Only
the refiner stack moves it (0.506). So a low PCK@1 here is a property of the
stride-4 coarse pipeline, not evidence that one encoder is blurrier than
another. Read PCK@5 for "did it match", PCK@1 for "how finely", EPE for the
tail.</li>
<li><b>All our arms are refiner-free, and that is the right comparison</b> for
the encoder question this project asks &mdash; but it means none of these EPEs
should be read as what a deployed matcher would achieve. The refiner headroom
(~1.2&nbsp;px all, ~1.2&nbsp;px far) sits on top of every row.</li>
<li><b>Reconstruction stays nearly free.</b> DINOv3 match&nbsp;&rarr;&nbsp;joint
costs 0.16 EPE all / 0.27 far; the deep 1-view joint arm is actually
<i>better</i> than its match control. So the ~6&nbsp;px deficit is attributable
to encoder depth, not to the reconstruction objective.</li>
<li><b>The v2 two-view decoder trades matching for appearance.</b> +0.30&nbsp;dB
and &minus;0.016 LPIPS against 1-view, but far EPE 12.92&nbsp;&rarr;&nbsp;14.64.
Note v2 also ran batch&nbsp;2&nbsp;&times;&nbsp;accum&nbsp;8 vs v1's
4&nbsp;&times;&nbsp;4 &mdash; same effective 16, not a byte-identical recipe.</li>
</ul>
<p>One caveat on the appearance axis: the DINOv3 arms have no <code>mv_vit</code>,
so their "mv tokens" are a linear map of desc and their post-hoc probe PSNR is
the 18.95&nbsp;dB desc ceiling by construction &mdash; which is why they sit on
the ceiling line in <code>fig_tradeoff.png</code>. <code>r2_v2</code> has no
probe yet.</p>

<h2>6. Where the models live</h2>
<p>Every arm named anywhere on this page, with the checkpoint that produced it.
Paths are relative to the repo root on the lab machine
(<code>/visinf/home/lab_mozkan/GenerativeRoMA</code>). <b><code>results/</code>
is gitignored</b> &mdash; these files exist on that machine only. Each directory
also holds <code>run.json</code> (the resolved config the run actually used),
<code>metrics.json</code>, <code>metrics_per_cat_co3d-unseen.json</code>, and
for decoder arms <code>recon_eval.json</code> + <code>recon_grid.png</code>.</p>
<div class="scroll">
<table>
  <caption>Checkpoints behind the figures. "figure label" is the x-axis label in
    fig_recon_quality.png / fig_generalization.png.</caption>
  <thead><tr><th>figure label</th><th>checkpoint</th><th>encoder</th>
    <th>steps</th><th>params</th><th>size</th></tr></thead>
  <tbody>
    <tr class="grp"><td colspan="6">from scratch, deep encoder (r2_v1 / r2_v2)</td></tr>
    <tr><td>deep match · v1 match</td>
      <td style="text-align:left"><code>results/r2_v1/match/ckpt.pt</code></td>
      <td style="text-align:left">mv_vit (ViT-B, 12 blocks)</td>
      <td>12k</td><td>117.2 M</td><td>1.3 GB</td></tr>
    <tr><td>deep joint 1v · v1 joint</td>
      <td style="text-align:left"><code>results/r2_v1/joint/ckpt.pt</code></td>
      <td style="text-align:left">mv_vit + RAEDecoder-B (view A) + GAN</td>
      <td>12k</td><td>203.6 M</td><td>2.4 GB</td></tr>
    <tr><td>deep joint 2v · v2 joint</td>
      <td style="text-align:left"><code>results/r2_v2/joint/ckpt.pt</code></td>
      <td style="text-align:left">mv_vit + RAEDecoder-B (views A+B) + GAN</td>
      <td>12k</td><td>203.6 M</td><td>2.4 GB</td></tr>
    <tr class="grp"><td colspan="6">from scratch, frozen DINOv3 desc + linear (r2_v1_desc)</td></tr>
    <tr><td>DINOv3 match</td>
      <td style="text-align:left"><code>results/r2_v1_desc/match/ckpt.pt</code></td>
      <td style="text-align:left">DescProj linear 2048&rarr;1024</td>
      <td>12k</td><td>31.9 M</td><td>0.4 GB</td></tr>
    <tr><td><b>DINOv3 joint</b></td>
      <td style="text-align:left"><code>results/r2_v1_desc/joint/ckpt.pt</code></td>
      <td style="text-align:left">DescProj + RAEDecoder-B + GAN</td>
      <td>12k</td><td>118.3 M</td><td>1.4 GB</td></tr>
    <tr><td>DINOv3 recon (pure RAE)</td>
      <td style="text-align:left"><code>results/r2_v1_desc/recon/ckpt.pt</code></td>
      <td style="text-align:left">DescProj + RAEDecoder-B, head skipped</td>
      <td>12k</td><td>118.3 M</td><td>1.2 GB</td></tr>
    <tr class="grp"><td colspan="6">fine-tuned from romav2.pt (r3_ft)</td></tr>
    <tr><td>ft match</td>
      <td style="text-align:left"><code>results/r3_ft/match/ckpt.pt</code></td>
      <td style="text-align:left">mv_vit, matcher lr 2e-5</td>
      <td>6k</td><td>117.2 M</td><td>1.3 GB</td></tr>
    <tr><td>ft joint</td>
      <td style="text-align:left"><code>results/r3_ft/joint/ckpt.pt</code></td>
      <td style="text-align:left">mv_vit + RAEDecoder-B, matcher lr 2e-5</td>
      <td>6k</td><td>203.6 M</td><td>2.3 GB</td></tr>
    <tr class="grp"><td colspan="6">released reference (no checkpoint of ours)</td></tr>
    <tr><td>pretrained (no refiners)</td>
      <td style="text-align:left"><code>~/.cache/torch/hub/checkpoints/romav2.pt</code></td>
      <td style="text-align:left">matcher.* keys only; results/r3_ft/pretrained/
        holds just its metrics</td>
      <td>&mdash;</td><td>117.2 M</td><td>1.0 GB</td></tr>
    <tr><td>pretrained + refiners</td>
      <td style="text-align:left"><code>~/.cache/torch/hub/checkpoints/romav2.0.1.pt</code></td>
      <td style="text-align:left">full released model, loaded by
        <code>RoMaV2()</code>; metrics in results/r3_ft/pretrained-refined/</td>
      <td>&mdash;</td><td>&mdash;</td><td>1.0 GB</td></tr>
  </tbody>
</table>
</div>
<p>Two loading gotchas. The <code>r2_v1_desc</code> checkpoints only load with
<code>desc_only=True</code> &mdash; their <code>mv_vit</code> keys are a
<code>DescProj</code> linear, not a ViT, so a default <code>R2Model</code>
raises on <code>load_state_dict</code>. And every arm reads the same frozen
descriptor cache
(<code>romav2_feats/r2_desc_320/&lt;cat&gt;_&lt;seq&gt;_&lt;frame&gt;.pt</code>,
fp16) rather than running DINOv3 itself, so a checkpoint alone is not runnable
on new images without the descriptor pass.</p>
<pre><code>from src.r2.model import R2Model
ckpt = torch.load("results/r2_v1_desc/joint/ckpt.pt", map_location="cpu",
                  weights_only=False)
model = R2Model(with_decoder=True, desc_only=True)   # desc_only is required
model.load_state_dict(ckpt["model"], strict=True)

# or through the entry point:
# python experiments/r3_eval_cats.py run=r2_v1_desc arm=joint step=12000 \
#     model.desc_only=true split=co3d-unseen tag=co3d-unseen</code></pre>

<p class="foot">Sources: <code>results/{{r2_v1,r2_v2,r2_v1_desc}}/*/metrics.json</code>
and <code>recon_eval.json</code>;
<code>results/&lt;run&gt;/&lt;arm&gt;/metrics_per_cat[_co3d-unseen].json</code>
(<code>experiments/r3_eval_cats.py arm=... split=...</code>); R1 probe numbers
from docs/R1.md; checkpoint metadata from each run's <code>run.json</code>.
Figures from <code>experiments/visualizations/r3_report_figs.py</code> and
<code>r3_pck_example_fig.py</code>; this page is rebuilt by
<code>experiments/visualizations/r3_matcher_report_html.py</code>.</p>
</main>
"""


def main():
    html = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Matcher results — GenerativeRoMA</title>\n"
        f"<style>{CSS}</style>\n</head>\n<body>\n{BODY}\n</body>\n</html>\n"
    )
    html = html.replace("{PCK}", b64(IMGS["pck"])).replace("{RECONQ}", b64(IMGS["reconq"])).replace(
        "{GEN}", b64(IMGS["gen"]))
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
