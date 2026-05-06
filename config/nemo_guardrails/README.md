# NeMo Guardrails Degree-3 Layout

This directory is now the active official NeMo Guardrails input configuration.

- `config.yml`: official NeMo config for the input/query safety stage
- `rails.co`: Colang flows for the active input rail
- `config.py`: registers project-specific safety actions as official NeMo custom actions

Additional active stage configs live in sibling directories:

- `config/nemo_guardrails_context`: official NeMo-backed context safety stage
- `config/nemo_guardrails_output`: official NeMo-backed output safety stage

The standard `SAFETY_BACKEND=nemo` path now uses the official `nemoguardrails.LLMRails`
runtime as the single active guardrails executor. Shared parsing utilities from
`src/core/safety_audit.py` are reused by NeMo custom actions, but there is no
separate legacy runtime path in the active request flow.
