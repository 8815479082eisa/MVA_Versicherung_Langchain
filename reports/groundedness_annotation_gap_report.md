# Groundedness Annotation Gap Report

## Current state

- Candidate cases: **498**
- Reviewer 1 completed labels: **0**
- Reviewer 2 completed labels: **0**
- Adjudicated labels: **0**
- Final locked hold-out: **not defined or locked**
- Groundedness scores exposed to reviewers: **0**

## Blocking conditions

Final calibration requires both independent reviews, conflict adjudication,
complete adjudicated PASS/FAIL labels for eligible cases, explicit error
categories and criticality, and a finalized leakage-safe locked hold-out.

**No final threshold calibration was performed because independent human annotation and adjudication are still pending.**

## Next action

1. Give each reviewer only their dedicated workbook sheet.
2. Run `scripts/analyze_groundedness_annotations.py` after both reviews.
3. Adjudicate every conflict and finalize the hold-out by leakage group.
4. Run isolated scoring only after adjudication.
5. Run extended calibration only after all readiness checks pass.
