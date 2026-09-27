# Mean Average Precision in Object Detection: Blind Spots and Alternative Metrics

[![CI](https://github.com/devtazi/rethinking-object-detection-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/devtazi/rethinking-object-detection-metrics/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![COCO](https://img.shields.io/badge/dataset-MS--COCO%202017-lightgrey.svg)](https://cocodataset.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A controlled study of mean Average Precision, the metric behind nearly every object
detection leaderboard. It does two things, in order:

1. **It criticises mAP**, through six experiments on MS-COCO in which the detector is
   known exactly by construction, so that a result can be attributed to the metric
   rather than to a model or a training run.
2. **It examines how far oLRP and TIDE are more relevant**, on the same detections,
   identifying which part of each metric carries the information mAP loses, and where
   those metrics fail in turn.

## Start here: two detectors and their scores

Before any parameter sweep, two pictures. Both are real COCO images, both show what a
detector predicted, and both come with the mAP that detector obtained over 1,000 images.

| ![Scenario B](docs/figures/scenario_b.jpg) | ![Scenario A](docs/figures/scenario_a.jpg) |
|:---:|:---:|
| **Scenario B.** Five boxes per object, four of them nowhere near it. | **Scenario A.** One box per object, each covering it fairly well. |
| **mAP@0.50 = 1.000** | **mAP@0.50 = 0.001** |
| The highest score the metric can award. | Indistinguishable from a detector that found nothing. |

A detector nobody would deploy is given a perfect score, and a detector whose every box
lands on its object is told it failed completely. The ordering the metric produces is
the reverse of the one a person looking at the images would produce.

**Why scenario B scores 1.000.** Each object receives one accurate box at high
confidence (IoU around 0.82, confidence in U(0.85, 0.95)) and four displaced boxes at low
confidence (IoU at most 0.497 by construction, confidence in U(0.10, 0.40)). This pattern has
a name in the literature, *spatial hedging*; it is defined and cited in experiment 3.
Average Precision ranks all detections by confidence and integrates precision against
recall. The four spurious boxes rank below every accurate box, so they only enter the
curve once recall has already reached 1 and there is no precision left to lose. They
cost nothing.

**Why scenario A scores 0.001.** Every box keeps its object's size and is shifted by 20%
of its width and height, which puts the IoU at exactly 0.471 for every object regardless
of its size. The COCO matching rule counts a prediction as correct only above IoU 0.50.
At 0.471 every prediction is therefore a false positive, and every object simultaneously
a false negative. The handful of true positives that remain are predictions that happen
to overlap a *different* object of the same class by more than 0.5.

Neither failure is a bug in the implementation: both are reproduced here against
`pycocotools` itself, and this project's from-scratch mAP agrees with it to 1e-9. They
are properties of the metric's definition.

These two scenarios are the starting point of the study. Each is a single configuration
chosen by hand: one shift value for A, one number of extra boxes for B.

Now, to move from the intuition provided by these experiments to the actual proof, Part 1
will consist of :

- **Experiment 1** states the problem the two scenarios share. It is not that these particular
  detectors are extreme, but that mAP gives the same score to detectors behaving in completely
  different ways, so no score can be read back to the behaviour that produced it.
- **Experiments 2 and 3** remove the hand-picking. Rather than one value each, they vary
  the parameter across its whole range and report mAP at every step: the shift of
  scenario A in experiment 2, the number of low-confidence boxes of scenario B in
  experiment 3. What the two pictures above show at one setting turns out to hold across
  the range.

This code accompanies the paper *"On the Relevance of Mean Average Precision in Object
Detection: A Controlled Experimental Study and Comparative Analysis of Alternative
Metrics"* (Tazi & Glissa, IMT Mines Alès & L2TI, Université Sorbonne Paris Nord, 2025).

## How a metric is judged here

Three criteria organise the critique. They are the framework of the accompanying paper
and of the oral presentation it is drawn from; the individual shortcomings they group
together are those enumerated for AP by Oksuz et al. [6] and, for redundant predictions,
by Jena et al. [3].

**Completeness.** A metric is complete when it accounts for the three ways a detector
can fail: imprecise localisation, false positives and false negatives. An incomplete
metric is silent about at least one of them, so a detector can degrade along that axis
without the score moving.

**Interpretability.** A metric is interpretable when its value indicates what is wrong.
This is the criterion mAP fails most plainly: mAP is a single number that carries no
information about the origin of the error, so a low mAP tells a practitioner that the
detector is bad and nothing about what to change. The same is true of a high one.

**Practicality.** A metric is practical when it survives deployment. It should behave
sensibly on rare categories and small validation sets, and it should help set the
confidence threshold at which the detector will actually run - which, as Oksuz et al.
note, is not optional: *"in a practical application, the detections are usually required
to be filtered owing to response time limitations"* [6].

mAP is examined against each in turn, then oLRP and TIDE are put through the same three.

## Method

Predictions are not produced by a trained detector. They are derived from MS-COCO
ground-truth boxes by seeded perturbation functions ([`scenarios.py`](src/mapstudy/scenarios.py)),
so every result traces back to a known prediction pattern and reproduces exactly. Where
a detector needs a knob set to a particular value, the knob is *solved for* by bisection
rather than chosen by hand ([`equivalence.py`](src/mapstudy/equivalence.py)).

Three synthetic detectors recur throughout, and one more family is introduced in
experiment 1:

| Name | Construction | Used in |
|---|---|---|
| **Scenario A** | every box shifted by 20% of its size, high confidence | opening, exp. 2, 8 |
| **Scenario B** | one accurate box plus four low-confidence displaced boxes per object | opening, exp. 3 |
| **Baseline** | every coordinate jittered by up to 25% of the box size, confidence drawn from U(0.1, 1.0) independently of the IoU, so the ranking is uninformative | exp. 5, 6, 10 |

Unless stated otherwise, results use the first 500 annotated images of COCO 2017 train
(3,552 objects) with seed 42. The reference table at the end of this README uses the
first 1,000 images (7,538 objects), as do the two figures above.

## The metrics compared

mAP is assumed known. Two alternatives are reported beside it from Part 1 onwards, and
the minimum needed to read the tables is the following.

- **oLRP** is an *error* in [0, 1], zero for a perfect detector, and it is the minimum of
  LRP over the confidence threshold. **τ\*** is the threshold attaining that minimum.
  The score is reported with three components: localisation, false positive and false
  negative.
- **TIDE** returns no score. For each of six error types it returns the AP that would be
  recovered if that error type alone were corrected, so a value of `Loc = 0.495` means
  that repairing localisation and nothing else would return 0.495 of AP.

The formula for LRP, the rule assigning each TIDE error type and the implementations used
are in [Metric definitions](#metric-definitions) at the end of this README.

---

# Part 1 - The case against mAP

## 1. mAP cannot distinguish detectors that fail in opposite ways

![Metric values for three detectors calibrated to the same mAP](docs/figures/equivalence.png)

The opening showed two detectors whose scores are misleading. This experiment shows the
structural reason: many different detectors share one score, so the score cannot be read
backwards.

Three detectors are constructed on the same 3,552 objects. Each commits exactly one of
the three error types and none of the others:

- **Missed objects.** Reports a subset of the objects and nothing else. Every box it
  emits is the ground-truth box itself, so it never produces a false positive; its only
  error is the objects it leaves out.
- **Spurious boxes.** Reports every object exactly, then adds boxes placed three
  box-widths away, which overlap their own object not at all and any other object only
  by accident. It never misses an object and never misplaces one; its only error is the
  boxes it adds.
- **Mislocalised boxes.** Reports every object, but displaces a fraction of its boxes by
  20% of their size, which puts them at IoU 0.471 and below the matching threshold. It
  adds no box and omits no object; its only error is where the boxes sit.

Emitting the ground-truth box itself wherever a box is meant to be correct keeps
localisation noise out of the comparison: the mislocalised detector is the only one whose
boxes are not exact. Each detector has a single parameter governing how often it commits
its error, and that parameter is bisected until mAP@0.50 reaches 0.500.

| | Missed objects | Spurious boxes | Mislocalised boxes |
|---|---:|---:|---:|
| Parameter solved for | `detect_rate` = 0.4990 | `ghost_rate` = 1.4023 | `error_rate` = 0.3457 |
| Proportion affected | 48.3% of objects omitted | 1.40 extra boxes per object | 35.4% of boxes displaced |
| Detections emitted | 1,837 | 8,533 | 3,552 |
| Precision at confidence ≥ 0.05 | **1.000** | **0.416** | **0.646** |
| Recall at confidence ≥ 0.05 | **0.517** | **1.000** | **0.646** |
| **mAP@0.50** | **0.4999** | **0.4999** | **0.5003** |
| **mAP@[.50:.95]** | **0.4999** | **0.4998** | **0.4972** |

A detector with perfect precision, a detector with perfect recall, and a detector with
neither. mAP@0.50 separates them by **0.0004** and mAP@[.50:.95] by **0.0026**. Averaging
over ten IoU thresholds does not help here: every box is either exact, and so a true
positive at all ten, or well below 0.50, and so a false positive at all ten, which leaves
the same value to average ten times. The residual 0.0031 by which the mislocalised
detector's two mAP columns differ comes from the few displaced boxes that reach IoU 0.50
with a neighbouring object of the same class but not the thresholds above it.

The mapping from detector behaviour to score is therefore **not injective**. A given
leaderboard position is compatible with all three detectors, and the number alone does
not determine which one produced it.

This has a direct practical cost. Told only that a detector scores 0.500, a practitioner
cannot determine whether to work on recall, on precision or on box regression, and those
are three different projects. The score locates the detector on a scale without
identifying what limits it, which is the sense in which mAP carries no information for
the decision that normally follows a measurement. Oksuz et al. list this first among AP's
shortcomings: the *"inability to distinguish very different RP curves"* [6].

The reference illustration of the problem is Fig. 1 of the LRP paper, where three
sketched detectors all reach AP = 0.5: *"Despite these very different characteristics,
the APs of these differently-behaving detectors are exactly the same (AP=0.5)"* [6].
What is added here is that the equality is *solved for* rather than drawn, on real data,
with the error profiles known exactly. Reproduce with `mapstudy equivalence`.

## 2. mAP@0.50 measures localisation only as a threshold test, and mAP@[.50:.95] only above it

![Metrics as localisation degrades](docs/figures/sweep_shift.png)

This is scenario A turned into a sweep. Instead of one shift of 20%, every box is
shifted diagonally by a growing fraction of its size, so the IoU of *every* prediction is
known in closed form (`r²/(2-r²)` with `r = 1 - shift`) and falls from 1.0 to 0.09.

| IoU of every prediction | 1.00 | 0.82 | 0.68 | 0.57 | 0.52 | **0.47** | 0.32 | 0.09 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mAP@0.50 | 1.000 | 1.000 | 1.000 | 0.996 | 0.994 | **0.003** | 0.002 | 0.001 |
| mAP@[.50:.95] | 1.000 | 0.700 | 0.400 | 0.199 | 0.100 | **0.001** | 0.001 | 0.000 |
| oLRP ↓ | 0.000 | 0.355 | 0.639 | 0.869 | 0.968 | **0.999** | 0.998 | 0.999 |

**mAP@0.50 carries almost no information about localisation.** Across the eight sweep
points above the threshold, where the IoU falls from 1.000 to 0.516, it moves by 0.006 in
total, then collapses between two adjacent points. It reports which side of 0.50 the
boxes sit on and little else. The shift putting the IoU exactly at 0.50 is
`1 - sqrt(2/3) ≈ 0.1835`: a detector at `shift = 0.1834` scores about 1.0 and one at
`shift = 0.1836` about 0.0, and no person looking at the two sets of boxes could tell
them apart. Scenario A sits at `shift = 0.20`, just past the edge.

**mAP@[.50:.95] answers this above the threshold.** Averaging AP over the ten thresholds
from 0.50 to 0.95 makes the score sensitive to how far above 0.50 each box sits: over the
same eight points it moves by 0.900 and tracks the IoU closely. This is the standard
response to the criticism above, and on this range it is a valid one. The criticism in
this section is therefore directed at mAP@0.50, not at the COCO primary metric.

**It stops working below the threshold.** All ten thresholds are at or above 0.50, so a
box that falls under 0.50 fails every one of them. Across the seventeen sweep points
below the threshold, where the IoU continues to fall from 0.471 to 0.087, mAP@[.50:.95]
moves by **0.000**. A detector whose boxes are marginally too loose and one whose boxes
are nowhere near their objects receive the same score.

One limitation remains above the threshold: the response is a staircase of ten discrete
steps rather than a continuous function, so a difference in localisation smaller than one
threshold interval does not register at all.

This is the second shortcoming Oksuz et al. list, the *"lack of directly measuring
bounding box localization accuracy"* [6]. The oLRP column is included here only to show
that a continuous response above the threshold is possible; it is discussed in
experiment 8.

## 3. mAP ignores spatial hedging

![Metric values as redundant boxes are added per object](docs/figures/sweep_hedges.png)

**Spatial hedging** is the name Jena et al. give to a detector that covers its bets:
*"Spatial Hedging refers to hedged predictions which are spatially perturbed versions of
each other"* [3]. Rather than commit to one box per object, the detector surrounds each
object with displaced variants at low confidence, so that whichever one turns out to be
right, something was predicted there. The same paper states the consequence for the
metric directly: *"Low-confidence FPs do not affect AP"* [3]. Its companion notion,
*category hedging*, predicts several categories for one object; it is not studied here.

Scenario B is one point of this sweep, at four hedging boxes per object. Here that
number grows from 0 to 12, taking the detector from 3,552 to 46,176 detections for the
same 3,552 objects.

| Spurious boxes per object | 0 | 4 | 8 | 12 |
|---|---:|---:|---:|---:|
| mAP@0.50 | 1.000 | 1.000 | 1.000 | 1.000 |
| mAP@[.50:.95] | 0.700 | 0.700 | 0.700 | 0.700 |
| oLRP ↓ | 0.355 | 0.355 | 0.355 | 0.355 |
| TIDE, every error type | 0.000 | 0.000 | 0.000 | 0.000 |
| F1 at confidence ≥ 0.05 | 1.000 | 0.333 | 0.207 | 0.159 |

Not one ranking-based metric reacts, to three decimal places, while the detector emits
thirteen times more boxes. This confirms on COCO detection what Jena et al. report for
instance segmentation: AP *"does not penalize duplicate predictions in the high-recall
range"* [3].

**The absence of response extends to oLRP and TIDE.** All three evaluate a *ranking* and
are free to select where to cut it, so all three remain unchanged. Only a metric
evaluated at a fixed operating point responds: F1 falls from 1.000 to 0.159.

The LRP paper makes no contrary claim. Oksuz et al. motivate
oLRP by the two shortcomings quoted above, distinguishing RP curves and measuring
localisation, and nowhere present it as a remedy for false positives ranked below every
true positive. Spatial hedging is outside the problem oLRP was designed to solve, and
the measurement here agrees with the paper rather than contradicting it.

### When hedging becomes visible

![Metrics as spurious boxes stop being separable](docs/figures/sweep_hedge-score.png)

Keeping four spurious boxes per object and raising the top of their confidence range towards
the accurate boxes' 0.85-0.95:

| Highest confidence of a spurious box | 0.40 | 0.84 | **0.89** | 0.94 | 0.99 |
|---|---:|---:|---:|---:|---:|
| mAP@0.50 | 1.000 | 1.000 | **0.953** | 0.801 | 0.647 |
| oLRP ↓ | 0.355 | 0.355 | **0.451** | 0.536 | 0.597 |
| oLRP localisation ↓ | 0.178 | 0.178 | 0.178 | 0.178 | 0.178 |

Nothing moves until the two confidence populations overlap, after which every ranking
metric degrades at once. **What these metrics register is the separability of the two
confidence populations, rather than the number of redundant boxes the detector emits.**

## 4. mAP has a floor that duplication cannot cross

![Metric values as near-identical boxes are added to every object](docs/figures/duplication.png)

Experiment 3 remains open to one objection. The redundant boxes it adds are all less
confident than the accurate ones, so a detector emitting them may be described as poorly
calibrated rather than poorly evaluated, and a confidence threshold would remove them.
The construction used here excludes that reading: every object is covered by
near-identical copies whose confidence is drawn from **the same distribution as the
accurate box**, so the two populations cannot be separated by any threshold.

| Copies added per object | 0 | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|---:|
| Detections for 3,552 objects | 3,552 | 7,104 | 17,760 | 60,384 | **230,880** |
| Precision at confidence ≥ 0.05 | 1.000 | 0.500 | 0.200 | 0.071 | **0.031** |
| **mAP@0.50** | 1.000 | 0.860 | 0.799 | 0.744 | **0.740** |
| F1 at confidence ≥ 0.05 | 1.000 | 0.667 | 0.333 | 0.132 | **0.060** |

**65 boxes per object, 230,880 detections for 3,552 objects, 97% of them wrong, and
mAP@0.50 still reports 0.740.** The curve flattens after the first few copies and stops
falling.

The mechanism differs from the one described in experiment 3, and the difference is what
makes this result independent of calibration.

Each copy is displaced by at most 5% of the box size on each axis, so it overlaps its
object at an IoU of 0.82 or above and would qualify as a true positive on its own. The COCO matching rule
processes detections in decreasing order of confidence and assigns the object to the
first box that covers it; every later box covering the same object is then a false
positive. The true positive is therefore not drawn at random among the `k+1` candidates,
it is **the most confident of them**, that is the maximum of `k+1` draws from the common
distribution. A maximum increases with the number of draws:

| Copies added per object | 0 | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|---:|
| Mean confidence of the true positives | 0.519 | 0.680 | 0.844 | 0.948 | **0.986** |
| Mean confidence of the false positives | - | 0.363 | 0.445 | 0.545 | 0.633 |
| True positives among the 5% most confident detections | 100% | 97.5% | 90.7% | 73.3% | **50.0%** |

AP integrates precision against recall along that ordering, so what it measures is the
purity of the top of the ranking. Duplication raises the confidence of each true positive
and leaves the losing copies below it, since they lost the same draw. At 64 copies only
3.1% of the detections are correct, as the first table shows, yet 50.0% of the most
confident ones still are: full recall is reached while precision is near one half, and
the area under the curve remains large.

This bounds what duplication can do to the score. Adding copies degrades precision and
raises the confidence of the true positives at the same time, and the two effects partly
cancel, which is why **mAP settles on a floor instead of falling to zero**, whatever the
confidence of the duplicates.

Reproduce with `mapstudy duplication`.

## 5. mAP reads the ranking of confidences, never their values

The baseline detector's confidences are replaced by strictly increasing functions of
themselves. The boxes, the labels and the order of the detections are untouched; only the
numbers attached to them change.

| Confidence transform | Confidence range | mAP@0.50 | mAP@[.50:.95] | F1 at confidence ≥ 0.05 |
|---|---|---:|---:|---:|
| identity | [0.100, 1.000] | 0.643157 | 0.177631 | 0.772 |
| `s³` | [0.001, 0.999] | 0.643157 | 0.177631 | 0.642 |
| `√s` | [0.316, 1.000] | 0.643157 | 0.177631 | 0.772 |
| logistic | [0.008, 0.998] | 0.643157 | 0.177631 | 0.696 |
| **`0.10 + 0.02·s`** | **[0.102, 0.120]** | **0.643157** | **0.177631** | 0.772 |

Both mAP columns are identical across every row, to the six decimals shown and to machine
precision beyond them; the test suite asserts equality rather than approximate equality.
The F1 column, evaluated at a fixed threshold of 0.05, ranges from 0.642 to 0.772
depending on the transform.

The final row gives the practical reading. Its detector assigns every detection a
confidence between 0.102 and 0.120, which places all of them above a threshold of 0.05
and below any threshold above 0.12, so at either of those settings it returns every
detection it has or none at all. It obtains the same mAP as the identity row, whose confidences span
[0.100, 1.000] and allow a threshold to be placed anywhere. **A detector that is never
confident is indistinguishable, under mAP, from one that is confident where it should
be**, although only the second can be operated at a threshold.

Oksuz et al. state the property directly: *"AP is not confidence-score sensitive. Since
the sorted list of the detections is required to calculate AP, a detector generating
results in a limited interval will lead to the same AP"* [6]. The final row is such a
limited interval. The information needed to choose a confidence threshold is therefore
absent from mAP by construction, since mAP is invariant under every transformation that
would change which threshold is appropriate.

## 6. Interpolation raises AP by an unpredictable amount

![AP added by the monotone envelope, by category size](docs/figures/interpolation_bias.png)

Before integrating, COCO replaces each precision by the highest precision reached at any
greater recall. This monotone envelope can only raise AP. The question is by how much,
and whether that amount is a fixed offset.

Measured on the baseline detector, per category, pooled over 8 seeds:

| Objects in the category | 1-2 | 3-5 | 6-10 | 11-25 | 26-50 | 51-100 | 101+ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mean AP added | +0.000 | +0.010 | +0.031 | +0.031 | +0.031 | +0.028 | +0.018 |
| Standard deviation | 0.001 | 0.020 | **0.035** | 0.026 | 0.020 | 0.016 | **0.009** |
| Worst single category | +0.005 | +0.085 | +0.098 | **+0.114** | +0.106 | +0.063 | +0.049 |

The mean is roughly flat at about 3 AP points from 6 objects upwards. What changes with
category size is not the size of the correction but its **predictability**: the standard
deviation is four times larger for categories of 6-10 objects than for categories past
100, and an individual small category can be handed **11 AP points** by interpolation
alone. Since mAP weights every category equally, those swings pass straight into the
leaderboard number.

Two qualifications bound this result. First, the effect is not monotone in category
size: below three objects the curve has too few steps to contain dips worth filling, and
the bias vanishes, so the correction is largest for categories of moderate size rather
than for the smallest ones. Second, "AP without interpolation" is not a standard metric.
It is computed here to isolate the contribution of the envelope, and no claim is made
that the uninterpolated value is the correct one: the envelope exists to limit the
sensitivity of AP to small changes in the ranking.

---

# Part 2 - How far oLRP and TIDE are more relevant

Every number in this part is computed on the **same detections** as Part 1. Two
properties are assessed for each metric: whether it distinguishes detectors to which mAP
assigns the same score, and whether its output indicates which error should be corrected.
The absolute level of a metric is left aside, since oLRP is an error to be minimised and
mAP a score to be maximised, and the two are not directly comparable.

## 7. The decomposed components separate detectors that the scalar metrics do not

Return to the three detectors of experiment 1, all calibrated to mAP@0.50 = 0.500. The
spread column reports `max - min` across the three; a metric that does not distinguish
them has a spread of zero.

| | Missed objects | Spurious boxes | Mislocalised boxes | **Spread** |
|---|---:|---:|---:|---:|
| mAP@0.50 | 0.4999 | 0.4999 | 0.5003 | **0.0004** |
| mAP@[.50:.95] | 0.4999 | 0.4998 | 0.4972 | **0.0026** |
| oLRP ↓ | 0.501 | 0.558 | 0.492 | **0.066** |
| oLRP localisation ↓ | 0.000 | 0.000 | 0.002 | 0.002 |
| **oLRP false positive** ↓ | 0.000 | 0.540 | 0.320 | **0.540** |
| **oLRP false negative** ↓ | 0.501 | 0.042 | 0.348 | **0.459** |
| **TIDE Loc** | 0.000 | 0.004 | 0.495 | **0.495** |
| **TIDE Bkg** | 0.000 | 0.470 | 0.000 | **0.470** |
| **TIDE Miss** | 0.500 | 0.000 | 0.000 | **0.498** |
| TIDE verdict | `Miss` | `Bkg` | `Loc` | - |

**The oLRP scalar discriminates weakly on this family: its spread is 0.066, against
0.0004 for mAP@0.50.** The improvement is two orders of magnitude, yet the three
detectors remain within 0.07 of one another, and a ranking by oLRP alone would convey
little about the ways in which they differ.

The separation comes from the components. The false-positive and false-negative terms
spread by 0.540 and 0.459, and for each detector the dominant TIDE error corresponds to
the error it was constructed to commit. The advantage of oLRP over mAP therefore lies in
the three quantities it reports rather than in the scalar that summarises them.

Stated in this form the claim is narrower than the one usually made for oLRP. A
leaderboard reporting oLRP as a single column retains little of that advantage.

### What oLRP's components leave undistinguished

The `oLRP localisation` row remains near zero for all three detectors, including the one
whose only error is mislocalisation. This follows from the definition of the term rather
than from a defect in the implementation. The component measures the tightness of boxes
that have already been matched, and a box falling below the IoU threshold is not a loose
true positive but a false positive, which in turn leaves its object unmatched. oLRP
therefore represents mislocalisation as **0.320 of false positive and 0.348 of false
negative**, the same reading it would give a detector that emits one spurious box and
separately misses one object.

TIDE distinguishes the two cases, reporting `Loc = 0.495` with every other error type at
zero. It does so by asking a different question: how much AP would be recovered if one
error type were corrected. Correcting localisation recovers what correcting any other
type would not.

The two are therefore complementary. oLRP decomposes the error into three commensurable
terms at a single operating point; TIDE attributes the lost AP to six named causes. The
present experiment requires both.

## 8. oLRP responds continuously to localisation, and only above the threshold

Experiment 2's sweep, with both mAP variants shown against oLRP:

| shift | 0.000 | 0.050 | 0.100 | 0.150 | 0.175 | **0.200** | 0.300 | 0.600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| True IoU of every box | 1.000 | 0.822 | 0.681 | 0.566 | 0.516 | **0.471** | 0.325 | 0.087 |
| mAP@0.50 | 1.000 | 1.000 | 1.000 | 0.996 | 0.994 | **0.003** | 0.002 | 0.001 |
| mAP@[.50:.95] | 1.000 | 0.700 | 0.400 | 0.199 | 0.100 | **0.001** | 0.001 | 0.000 |
| oLRP ↓ | 0.000 | 0.355 | 0.639 | 0.869 | 0.968 | **0.999** | 0.998 | 0.999 |

Compared with **mAP@0.50**, the difference is large: over `shift ∈ [0, 0.175]`
localisation quality halves, mAP@0.50 moves by 0.006 and oLRP traverses 0.00 to 0.97
monotonically.

Compared with **mAP@[.50:.95]**, it is considerably smaller. Both metrics track the IoU
over that range, and they differ in granularity: oLRP varies continuously, whereas
mAP@[.50:.95] changes only when a box crosses one of ten fixed thresholds, so it moves in
steps of 0.1 and does not register a localisation difference smaller than 0.05 of IoU.
The advantage is genuine but limited.

**Below the threshold, neither metric is informative.** Past IoU 0.50 the true IoU
continues to fall from 0.471 to 0.087 while oLRP saturates at 0.999 and mAP@[.50:.95]
remains at 0.001. The localisation term of oLRP is normalised by `1 - 0.5` and counts
only matched true positives, so it grades a detector that is close to the threshold and
says little about one that is far from it. The limitation experiment 2 establishes for
mAP below the threshold applies equally to oLRP.

What holds over the whole range is the decomposition. oLRP reports localisation as a
**separate term**, so the result indicates how much of the error is attributable to loose
boxes rather than to missing or spurious ones, whereas mAP@[.50:.95] returns a single
value in which the three are already combined. This is the property measured in
experiment 7.

### The attribution TIDE adds to the score

![AP recoverable by fixing each error type](docs/figures/sweep_shift_tide.png)

At `shift = 0.20`, mAP reports 0.003 and provides nothing further. TIDE attributes
**0.983 of the 0.997 lost AP to localisation**, with every other error type at zero,
which identifies what should be corrected. On scenario A itself,
evaluated over 1,000 images at mAP@0.50 = 0.001, TIDE attributes **0.986 of the 0.999
lost AP to `Loc`**.

The figure also shows a limitation of TIDE, which is why the summary table below does not
credit it with covering localisation over the whole range. Below IoU = 0.1, its
background threshold, a prediction ceases to be a mislocalised box and becomes a spurious
one: the attribution to `Loc` falls from 0.884 to 0.053 and **no other error type takes
its place**. TIDE measures the AP recovered by correcting *one* error type at a time, and at
that point the detector is wrong in two respects simultaneously, so no single correction
recovers anything. Roughly 0.94 of the lost AP is then attributed to no cause at all.

## 9. oLRP continues to degrade under duplication after mAP has reached its floor

Experiment 4, with oLRP added:

| Copies added per object | 0 | 1 | 4 | 16 | 64 |
|---|---:|---:|---:|---:|---:|
| Detections | 3,552 | 7,104 | 17,760 | 60,384 | 230,880 |
| mAP@0.50 | 1.000 | 0.860 | 0.799 | 0.744 | **0.740** |
| **oLRP ↓** | 0.000 | 0.379 | 0.476 | 0.546 | **0.555** |
| oLRP false positive ↓ | 0.000 | 0.245 | 0.285 | 0.305 | 0.301 |
| **oLRP optimal threshold τ\*** | 0.211 | 0.471 | 0.763 | 0.925 | **0.981** |
| TIDE `Dupe` | 0.000 | 0.136 | 0.189 | **0.240** | 0.239 |
| F1 at confidence ≥ 0.05 | 1.000 | 0.667 | 0.333 | 0.132 | 0.060 |

This is the largest advantage the oLRP scalar shows anywhere in the study. The two
metrics are not on a common scale, so what is compared is the shape of the response: mAP
reaches a plateau after the first few copies and remains there, while oLRP continues to
change over the whole range. The contrast with experiment 3 has a precise cause. There the redundant boxes were less
confident than every accurate one, so the minimisation over thresholds that defines oLRP
excluded them at no cost. Here they are drawn from the same distribution as the accurate
boxes, no threshold separates the two populations, and oLRP therefore counts them.

The false-positive term stops rising between 16 and 64 copies, at 0.305 and 0.301. It is
evaluated at τ\*, which by then discards almost every copy, so the term describes the
detections kept at that threshold rather than the total emitted. The count of redundant
boxes is carried by τ\* itself, not by this component.

The optimal threshold τ\* is not an auxiliary output of the metric. oLRP is defined as
`min_τ LRP(τ)`, so computing the score requires locating the threshold at which the
minimum is attained; τ\* is that argmin, obtained together with the score rather than in
addition to it. Oksuz et al. state this as the purpose of the construction: *"Optimal LRP
determines the 'best' confidence score threshold for a class, which balances the
trade-off between localization and recall-precision"* [6]. Selecting an operating point
is therefore not an application of oLRP but part of its definition.

Here τ\* supplies exactly what the score does not. As duplication increases it rises from
0.211 to 0.981, so the best operating point available lies above 0.98 and every detection
below that confidence has to be discarded. mAP admits no comparable output, since experiment 5 establishes that it is
invariant under the transformations that would change the answer. τ\* is exposed by
[`evaluate_lrp`](src/mapstudy/evaluation.py); the values reported here are means over the
categories, the threshold being determined per class in the original formulation.

## 10. oLRP is equally rank-invariant, while τ\* is not

Experiment 5, with oLRP added. This is the comparison in which oLRP's advantage is
smallest:

| Confidence transform | Confidence range | mAP@0.50 | oLRP ↓ | **τ\*** | F1 at confidence ≥ 0.05 |
|---|---|---:|---:|---:|---:|
| identity | [0.100, 1.000] | 0.643157 | 0.8108 | **0.288** | 0.772 |
| `s³` | [0.001, 0.999] | 0.643157 | 0.8108 | **0.074** | 0.642 |
| `√s` | [0.316, 1.000] | 0.643157 | 0.8108 | **0.507** | 0.772 |
| logistic | [0.008, 0.998] | 0.643157 | 0.8108 | **0.191** | 0.696 |
| `0.10 + 0.02·s` | [0.102, 0.120] | 0.643157 | 0.8108 | **0.106** | 0.772 |

**The oLRP score is invariant as well**, to the four decimal places shown, for the same
reason as mAP: it is a minimum over thresholds, and a monotone remapping displaces the
thresholds together with the confidences. The table is therefore a counterexample to the
claim, sometimes made, that oLRP accounts for confidence values where mAP does not.

What does depend on those values is τ\*, which moves from 0.074 to 0.507 across the
rows. The invariance therefore affects the score and not the threshold: oLRP scores the
detector identically under all five transforms, as mAP does, while reporting a different
operating point for each of them. That output has no counterpart in mAP, which returns a
score alone.

## 11. The three criteria, reviewed

Each cell cites the experiment it rests on, so that no claim in the table extends beyond
the measurements reported above.

| | | mAP | oLRP | TIDE |
|---|---|---|---|---|
| **Completeness** | Localisation | mAP@0.50 binary at the threshold; mAP@[.50:.95] tracks IoU above it in ten steps, blind below (exp. 2) | Continuous above the threshold, saturated below, and reported as a separate term (exp. 8) | Attributes lost AP to `Loc` above IoU 0.1, nothing below it (exp. 8) |
| | FP vs FN | Merged into one curve, indistinguishable (exp. 1) | Separated: spreads 0.54 and 0.46 (exp. 7) | Separated, and by *kind* of FP: `Bkg`, `Dupe`, `Cls` (exp. 7) |
| | Redundant boxes, high confidence | Floor at 0.740 under 65x duplication (exp. 4) | Degrades from 0.000 to 0.555 (exp. 9) | `Dupe` rises to 0.24 (exp. 9) |
| | Spatial hedging, low confidence | No response (exp. 3) | **No response** (exp. 3) | **No response** (exp. 3) |
| **Interpretability** | What the score indicates should be corrected | Nothing: three detectors, one score (exp. 1) | Which of three error terms dominates (exp. 7) | Which of six named causes, ranked by AP recoverable (exp. 7) |
| | As a single number | The case under examination: spread 0.0004 (exp. 7) | Spread 0.066, of the same order as mAP (exp. 7) | Not a single number by construction |
| **Practicality** | Choosing a threshold | Not possible: the score is invariant under every remapping that would change the answer (exp. 5) | **Determined by construction**: oLRP is `min_τ LRP(τ)`, so τ\* is the argmin obtained with the score. It tracks the confidence values where the score does not (exp. 9, 10) | - |
| | Rare categories | Interpolation adds up to +0.11 AP, unpredictably (exp. 6) | Not affected: oLRP integrates no precision-recall curve | Reuses COCO's interpolated AP and inherits the same bias (exp. 6) |

## Conclusion

The experiments support two claims about mAP and one qualified claim about its
alternatives.

**mAP is not injective.** Detectors with opposite precision and opposite recall reach the
same score, and averaging AP over ten IoU thresholds does not separate them. mAP is
further invariant under every monotone transformation of the confidences, so it does not
determine the confidence threshold at which a detector should be deployed. In both cases
the metric orders detectors without characterising them: a low mAP indicates that
performance is poor without indicating in what respect, and a high mAP, as the opening
pair and experiments 3 and 4 show, may be obtained by a detector that could not be used.

**The components of oLRP and the error types of TIDE recover what mAP combines.** On the
same detections, they distinguish detectors that mAP scores identically, and τ\* supplies
an operating point that mAP does not express. This corresponds to the *"richer and more
discriminative information than AP"* claimed for oLRP by Oksuz et al. [6], and
experiments 1 and 7 measure it directly. Reporting oLRP with TIDE is an improvement over
reporting mAP alone.

**The stronger claim, that oLRP is a better scalar than mAP, is not supported here.** As
a single number, oLRP separates the three calibrated detectors by 0.066, does not respond
to low-confidence hedging, is invariant under confidence remapping, and saturates below
the IoU threshold. The improvement measured in this study is attributable to the
decomposition, which reports three quantities where mAP reports one.

The resulting recommendation follows from the limitation the three metrics share: report
the components of oLRP together with τ\* and a TIDE breakdown, and add one metric
evaluated at a fixed operating point. The fixed-threshold F1 is the only quantity in this
study that responded to every failure mode tested.

---

# Reference results

The three detectors of the Method table, evaluated with every metric on the first 1,000
annotated images of COCO 2017 train (7,538 objects), seed 42. These are the numbers the
opening pair is taken from. Reproduce with `mapstudy run --all`.

| Metric | Baseline | Scenario A | Scenario B |
| --- | ---: | ---: | ---: |
| mAP@[.50:.95] (pycocotools) | 0.166 | 0.000 | 0.700 |
| mAP@.50 (pycocotools) | 0.638 | 0.001 | 1.000 |
| mAP@.50 (ours, COCO 101-point) | 0.638 | 0.001 | 1.000 |
| mAP@.50 (ours, VOC all-point) | 0.639 | 0.001 | 1.000 |
| mAP@.50 (ours, no interpolation) | 0.611 | 0.001 | 1.000 |
| oLRP ↓ | 0.814 | 0.999 | 0.355 |
| oLRP localisation ↓ | 0.354 | 0.398 | 0.178 |
| oLRP false positive ↓ | 0.224 | 0.903 | 0.000 |
| oLRP false negative ↓ | 0.233 | 0.990 | 0.000 |
| TIDE AP@.50 | 0.638 | 0.001 | 1.000 |
| TIDE dominant error | Loc | Loc | none |
| Detections / objects | 7538 / 7538 | 7538 / 7538 | 37690 / 7538 |

Three observations on this table.

**The from-scratch mAP agrees with `pycocotools` to three decimals** on all three
detectors, and the test suite checks the agreement to 1e-9. The behaviour reported in
this README is therefore a property of the metric rather than of a particular
implementation.

**Scenario B's mAP@[.50:.95] = 0.700 follows from its construction.** The accurate boxes
sit at an IoU of about 0.82, so they are true positives at the seven thresholds from 0.50
to 0.80 and false positives at the three above, giving seven tenths of the maximum.

**Neither oLRP nor TIDE responds to scenario B's hedging.** oLRP reports 0.355, but its
false-positive term is **0.000**: the value is due entirely to the residual localisation
error of the accurate boxes. TIDE reports no error to correct, since it measures the AP
that a correction would recover and there is none to recover. This is the limitation
established in experiment 3, observed here on the scenario that motivates the study.

# Metric definitions

Referenced from [The metrics compared](#the-metrics-compared) in the front matter. mAP is
assumed known; the two alternatives are defined in full below.

## oLRP (Oksuz et al. [6])

LRP is an **error**, so lower is better, and it is bounded in [0, 1] with 0 for a perfect
detector. It is evaluated at a confidence threshold τ, which fixes which detections are
kept and therefore the counts of true positives, false positives and false negatives:

```
LRP(τ) = [ Σ_TP (1 - IoU) / (1 - 0.5) + FP(τ) + FN(τ) ] / [ TP(τ) + FP(τ) + FN(τ) ]
```

The numerator adds three quantities on a common scale: the looseness of the matched
boxes, normalised by `1 - 0.5` so that a box exactly at the IoU threshold contributes 1;
the number of false positives; and the number of false negatives. The denominator is the
total number of decisions made, which turns the sum into a rate.

**Optimal LRP** is the minimum of that error over every threshold, and **τ\*** is the
threshold attaining it:

```
oLRP = min_τ LRP(τ)        τ* = argmin_τ LRP(τ)
```

Two consequences are used throughout. Computing oLRP requires locating τ\*, so the
operating point is produced by the definition rather than added to it. And because oLRP
is evaluated at the threshold that suits the detector best, detections that any threshold
can discard cost it nothing, which is what experiment 3 measures.

The score is reported with **three components**, each read at τ\*. `oLRP localisation`
is the mean of `1 - IoU` over the matched boxes, reported without the `1 - 0.5`
normalisation that appears in the formula above: scenario B, whose accurate boxes sit at
IoU 0.822, gives 0.178. `oLRP false positive` is `1 - precision` and `oLRP false
negative` is `1 - recall`, both at τ\*. Part 2 shows that these three carry the
information the scalar does not.

## TIDE (Bolya et al. [1])

TIDE does not return a score. For each of six error types it returns **the AP that would
be recovered if that error type alone were corrected**, every other error left in place.
A value of `Loc = 0.495` therefore means that repairing localisation, and nothing else,
would return 0.495 of AP. Two properties follow: the six values do not have to sum to the
AP that was lost, and a detector can lose AP with every error type at zero, which is what
happens in scenario B.

With the default thresholds of 0.5 for a match and 0.1 for background, a detection that
failed to match is assigned the first of these that applies:

| Error | A detection that is not a true positive because | Correcting it means |
|---|---|---|
| `Loc` | its IoU with the best same-class object is between 0.1 and 0.5 | tightening the box |
| `Cls` | it reaches IoU 0.5 on an object of *another* class | relabelling it |
| `Dupe` | it reaches IoU 0.5 on a same-class object already matched by a more confident detection | suppressing it |
| `Bkg` | its IoU is at most 0.1 with *every* object, of any class | not predicting it |
| `Both` | none of the above applies: wrong class and insufficient IoU together | both of the above |
| `Miss` | an object matched nothing, and no correction above would have recovered it | detecting it |

The order matters: `Both` is the case left over once the others are excluded, and `Miss`
counts only objects that no single correction could recover, so it is narrower than
"object without a detection".

## Fixed-threshold precision, recall and F1

Computed at one confidence threshold, 0.05 unless stated. Unlike the three metrics above,
these do not rank detections; they count every detection kept by the threshold. This is
the only quantity in the study that responded to every failure mode tested.

## Implementations

| Metric | Implementation |
|---|---|
| mAP@.50, mAP@[.50:.95] | [`pycocotools`](https://github.com/cocodataset/cocoapi) and [`average_precision.py`](src/mapstudy/average_precision.py), which exposes the interpolation scheme and is tested to match `pycocotools` to 1e-9 |
| oLRP, its components and τ\* | [`third_party/cocoeval_lrp.py`](src/mapstudy/third_party/cocoeval_lrp.py), the COCO evaluator extended with LRP |
| TIDE | [`tidecv`](https://github.com/dbolya/tide). Its AP is COCO's interpolated AP, so it inherits the bias measured in experiment 6 |
| Precision, recall, F1 | [`average_precision.py`](src/mapstudy/average_precision.py) |

# Getting started

```bash
git clone https://github.com/devtazi/rethinking-object-detection-metrics.git
cd rethinking-object-detection-metrics
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

COCO 2017 annotations are read from the [`HichTala/coco`](https://huggingface.co/datasets/HichTala/coco)
Parquet export on the Hugging Face Hub. Only the shards that are needed are downloaded
and cached: the first one (about 475 MB) covers roughly 3,000 images.

```bash
# the opening pair and the reference table
mapstudy run --all                # baseline, scenario A, scenario B; writes results/
mapstudy visualize --scenario b --image-index 5

# Part 1, the case against mAP
mapstudy equivalence              # exp. 1 and 7: three detectors, one mAP
mapstudy sweep shift              # exp. 2 and 8: scenario A swept, plus the TIDE breakdown
mapstudy sweep hedges             # exp. 3: scenario B swept
mapstudy sweep hedge-score        # exp. 3: when hedging becomes visible
mapstudy duplication              # exp. 4 and 9: the floor duplication cannot cross
mapstudy invariance               # exp. 5, 6 and 10: score invariance, interpolation bias

pytest                            # offline, no dataset download
```

[`notebooks/walkthrough.ipynb`](notebooks/walkthrough.ipynb) walks through the scenarios,
builds a precision-recall curve step by step and compares the metrics.

# Repository layout

```
src/mapstudy/
├── boxes.py               IoU and box geometry
├── data.py                Data structures and COCO loading
├── scenarios.py           Synthetic detectors: baseline, A, B, and the equivalence family
├── average_precision.py   From-scratch AP / mAP and fixed-threshold operating points
├── evaluation.py          pycocotools, custom mAP, oLRP (with τ*) and TIDE
├── sweep.py               Parameter sweeps (experiments 2, 3, 8)
├── equivalence.py         Calibration to a common mAP, and the duplication floor (1, 4, 7, 9)
├── invariance.py          Score invariance and interpolation bias (5, 6, 10)
├── reporting.py           Results table
├── visualization.py       Figures
├── cli.py                 `mapstudy` command
└── third_party/           Vendored COCO evaluator extended with LRP
tests/                     Unit tests, including a cross-check against pycocotools
results/                   Reference outputs, one JSON per experiment
docs/figures/              Figures used in this README
```

# Limitations

- The synthetic detectors produce isolated failure modes. Real detectors combine several
  error types, and the calibration procedure of experiment 1 requires a model whose
  errors can be varied independently, which a trained detector does not provide.
- The first images of the train split are used rather than a random sample, which may
  bias the category distribution.
- Experiment 1 calibrates on mAP@0.50. The three detectors also agree on mAP@[.50:.95] to
  within 0.003, but this is an observed consequence of their construction rather than an
  imposed constraint.
- The operating point is reported at a single confidence threshold (0.05), chosen
  arbitrarily. An analysis of precision and recall across the range of thresholds would
  be more informative.
- mAP, oLRP and TIDE rely on the same one-to-one matching between predictions and ground
  truths. In crowded scenes the matching can itself determine the result, which accounts
  for the small number of true positives observed in scenario A.
- All results are computed at bounding-box level.

# References

1. Bolya, D., Foley, S., Hays, J., & Hoffman, J. (2020). [TIDE: A General Toolbox for Identifying Object Detection Errors](https://dbolya.github.io/tide/). ECCV.
2. Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., & Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) Challenge. *IJCV*, 88(2), 303-338.
3. Jena, R., Zhornyak, L., Doiphode, N., Chaudhari, P., Buch, V., Gee, J., & Shi, J. (2023). [Beyond mAP: Towards Better Evaluation of Instance Segmentation](https://arxiv.org/abs/2207.01614). CVPR.
4. Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). Panoptic Segmentation. CVPR.
5. Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., & Zitnick, C. L. (2014). [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312). ECCV.
6. Oksuz, K., Cam, B. C., Akbas, E., & Kalkan, S. (2018). [Localization Recall Precision (LRP): A New Performance Metric for Object Detection](https://arxiv.org/abs/1807.01696). ECCV.
7. Oksuz, K., Cam, B. C., Kalkan, S., & Akbas, E. (2021). Imbalance Problems in Object Detection: A Review. *IEEE TPAMI*, 43(10), 3388-3415.
8. Padilla, R., Passos, W. L., Dias, T. L. B., Netto, S. L., & da Silva, E. A. B. (2021). A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit. *Electronics*, 10(3), 279.
9. Rezatofighi, H., Tsoi, N., Gwak, J., Sadeghian, A., Reid, I., & Savarese, S. (2019). [Generalized Intersection over Union](https://arxiv.org/abs/1902.09630). CVPR.
10. Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP System Demonstrations.

# Authors

Adam Tazi and Mohamed Glissa, IMT Mines Alès.
Supervised by Hajer Fradi and Hicham Talaoubrid from Université Sorbonne Paris Nord, L2TI laboratory.
