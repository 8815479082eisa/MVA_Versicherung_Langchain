# NeMo Guardrails Degree 3

## Current State

The standard `SAFETY_BACKEND=nemo` path now uses the official NeMo Guardrails
runtime (`LLMRails` / `RailsConfig`) as the single active guardrails executor.

- Input safety runs through `config/nemo_guardrails`
- Context safety runs through `config/nemo_guardrails_context`
- Output safety runs through `config/nemo_guardrails_output`

Each stage is backed by an official NeMo configuration directory and an
official `config.py` extension hook. Shared parsing utilities from
`src/core/safety_audit.py` are reused inside the NeMo custom actions in
`src/guardrails/integrations/nemo_actions.py`.

## Architectural Consequence

The old project-specific runtime and flow engine have been removed from the
active codebase. The active call path is now:

`FastAPI -> rag_service -> NemoSafetyChecker -> official LLMRails stage config -> RAG pipeline`

## Remaining Limitation

Context safety is integrated through an official NeMo-backed stage-specific
configuration, but it is not a fully native NeMo retrieval rail in the sense of
NeMo owning the retrieval process itself. This is a consequence of the existing
external RAG orchestration in `src/api/rag_service.py`, where retrieval still
happens outside of NeMo. The current implementation therefore uses official
NeMo execution with custom context actions as the best realistic integration for
the current architecture.
