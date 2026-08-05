# Extended Groundedness Dataset - Duplicate and Leakage Report

## Grouping method

Initial families were defined from source question, document/page, customer,
policy/version and synthetic template relationships. Near-duplicate pairs were
then detected with TF-IDF word unigram/bigram cosine similarity >=
0.94. Any connected initial families were unioned
before split assignment. No case-level random split was used.

## Results

- Final leakage groups: **149**
- Near-duplicate pairs: **319**
- Leakage groups appearing in multiple splits: **0**
- Near-duplicate pairs crossing splits: **0**
- Calibration candidates: **277**
- Validation candidates: **88**
- Locked hold-out candidates: **133**

The hold-out assignment is preliminary and must not be finalized until human
annotation is complete. Generator expectations were used only to balance this
planning split; they are not Ground Truth.

## Known leakage risks

1. InsuranceQA questions can be semantically related even below the 0.94 threshold.
2. CRM variants for one customer may be numerous; all are kept in one customer group.
3. All Lara old/current-policy variants are kept together.
4. Active Helvetia and excluded Baloise source families are explicitly identified; excluded documents are never treated as current-product PASS evidence.
5. The 200-row thesis InsuranceQA subset is fully contained in the 1,000-row source and was not sampled independently.
