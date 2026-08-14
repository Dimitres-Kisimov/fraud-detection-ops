"""The one place the figure palette is defined.

The subject is a fraud review floor, so the figures are drawn the way its paper
trail looks: warm neutrals for everything routine, ALERT AMBER for the alert /
review-queue stream, CONFIRMED-FRAUD RED for loss and for the "this has moved"
lines, and CLEARED GREEN used sparingly for the state an analyst reaches at the
end of a review. No decorative hues, and no wall of technical blue.

VALIDATED, NOT EYEBALLED. The three series colours were checked with the
data-viz palette validator on the light print surface (#fcfcfb), all pairs:

    validate_palette "#a06a00,#9c1c1c,#3fae63" --mode light --surface "#fcfcfb" --pairs all
      [PASS] Lightness band       all 3 inside L 0.43-0.77
      [PASS] Chroma floor         all 3 >= 0.1
      [PASS] CVD separation       worst all-pairs #3fae63 <-> #a06a00 dE 10.8 (deutan)
      [PASS] Normal-vision floor  worst all-pairs #9c1c1c <-> #a06a00 dE 16.8 (normal)
      [WARN] Contrast vs surface  below 3:1 - relief required: #3fae63 at 2.75
      -> ALL CHECKS PASS

Red / amber / green is the traffic-light triple, which is exactly the set that
normally collapses under red-green colour blindness: the naive picks measure a
deuteranopia dE of 2.6 (against a floor of 8), i.e. indistinguishable. These
three survive because they are separated by LIGHTNESS as well as hue (OKLCH L
0.45 red, 0.57 amber, 0.66 green), which is what the validator run above
measures. Do not "brighten" the red or "deepen" the green without re-running it.

RELIEF RULE: cleared green sits at 2.75:1 on the light surface, below the 3:1
mark threshold, so every mark drawn in it also carries a visible direct label
(bar value, legend entry, or annotated point) - never colour alone. The same
rule covers the sub-3:1 neutral fills.
"""

from __future__ import annotations

# --- paper trail: surfaces, ink and chrome ---------------------------------
SURFACE = "#fcfcfb"     # chart surface (paper)
INK = "#0b0b0b"         # primary ink
INK_2 = "#52514e"       # secondary ink
MUTED = "#898781"       # axis and tick labels
GRID = "#e1e0d9"        # hairline gridlines
BASELINE = "#c3c2b7"    # axis lines, reference rules

# --- series ----------------------------------------------------------------
ALERT_AMBER = "#a06a00"    # the alert / review-queue stream, and its cost
FRAUD_RED = "#9c1c1c"      # confirmed fraud, loss, and "this has shifted"
CLEARED_GREEN = "#3fae63"  # cleared / calibrated / the state after review (sparing)

# --- neutral fills used for comparators (always with direct labels) --------
CASE_GRAPHITE = "#6b6459"  # quiet, in-focus neutral (markers, flat monitors)
PAPER_STONE = "#b0a999"    # recessive neutral (the strategies being beaten)
