# Architecture Guide - Weld Defect Vision

Updated: 2026-05-30

Use this page as the short path through the repository. It keeps the architecture grounded in the code, docs, commands, and boundaries that are already present.

## Summary

| Field | Notes |
|---|---|
| Lane | B2B industrial AI validation |
| Core idea | YOLOv8 defect workflow with model governance, serving, and operator-readable evidence. |
| Primary reader | Manufacturing quality teams, welding inspection groups, industrial AI, and edge deployment architecture inspection paths. |
| Stack | Python, Docker |

## Open First

1. Start with the README fast path and architecture section.
2. Open `docs/service-launch-playbook.md` only when architectureing the product or service angle.
3. Check the commands below before making claims about quality.
4. Skim the CI workflows and fixture data before deeper implementation architecture.
5. Read the boundaries section before presenting the project externally.

## Checks

| Purpose | Command |
|---|---|
| Test suite | `python -m pytest` |

## CI

- .github/workflows/architecture-blueprint.yml
- .github/workflows/ci.yml
- .github/workflows/dependency-architecture.yml
- .github/workflows/export-onnx.yml
- .github/workflows/repository-health.yml
- .github/workflows/repository-surface.yml
- .github/workflows/secret-scan.yml

## Evidence

- pytest/ruff-style local verification path
- containerized delivery path
- pytest passes
- Model card exists
- API detection path is documented

## Architecture Notes

| Possible offer | Working scope assumption |
|---|---|
| Inspection PoC | Scope after product intake |
| Model validation study | Scope after product intake |
| Edge-serving readiness assessment | Scope after product intake |

## Boundaries

- Human inspection required
- Site-specific validation
- No production acceptance without criteria

## Useful Metrics

- mAP/recall by defect
- False-negative architecture
- Operator architecture time
