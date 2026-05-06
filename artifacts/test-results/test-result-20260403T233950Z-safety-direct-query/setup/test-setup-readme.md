

# Test Setup: test-setup-safety-direct-query-v1

## Objective
Validate whether the safety layer detects suspicious direct user queries at input stage and allows benign queries.

## Scope
This setup tests only the input stage of the safety layer.

## Test Data
The test cases are stored in:
`data/benchmarks/safety/direct_query_attacks.jsonl`

## Test Categories
- direct_injection
- policy_bypass
- role_override
- benign_control

## Expected Behavior
- Suspicious queries should be flagged, blocked, or sanitized.
- Benign queries should be allowed.

## Notes
This is the first safety-specific validation step and focuses only on direct user input.