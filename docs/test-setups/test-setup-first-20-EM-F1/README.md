# test-setup-first-20-EM-F1

## Dataset
InsuranceQA

## Split
test

## Sample Size
20

## Metrics
- Exact Match (EM)
- Token F1

## Evaluation Script
scripts/eval_insuranceqa.py

## Run Parameters
- api-url: http://localhost:8000/api/ask
- split: test
- max-samples: 20
- count: 20
- timeout: 3600

## Purpose
Preliminary small-scale evaluation for the real RAG-based pipeline.

## Notes
- This is a preliminary 20-sample test setup.
- This is not the official 1k setup.