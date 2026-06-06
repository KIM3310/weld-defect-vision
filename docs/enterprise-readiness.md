# Enterprise Readiness Notes - Weld Defect Vision

Updated: 2026-05-30

This note defines what an enterprise buyer, public-sector reviewer, serious user, or technical evaluator can safely infer from this repository today. It is intentionally conservative: public proof is separated from production claims.

## Scope

| Field | Notes |
|---|---|
| Repository | `weld-defect-vision` |
| Lane | B2B industrial AI validation |
| Primary reader or buyer | Manufacturing quality teams, welding inspection groups, industrial AI, and edge deployment reviewers. |
| Core wedge | YOLOv8 defect workflow with model governance, serving, and operator-readable evidence. |
| Stack | Python, Docker |
| Readiness posture | Pilot-ready technical surface; production use requires customer-specific identity, monitoring, data, and support controls. |

## Enterprise Controls

| Control | Current expectation |
|---|---|
| Data boundary | Public artifacts should use demo, fixture, or synthetic data until the buyer approves data handling, retention, and access controls. |
| Identity and access | Production pilots should add SSO/OIDC, RBAC, scoped service accounts, secret rotation, and admin-visible access reviews. |
| Auditability | Keep decision logs, generated reports, CI results, eval outputs, and operator handoff artifacts reviewable. |
| Observability | Track health checks, latency, error budget, cost, eval pass rate, audit-log completeness, and handoff/report generation status. |
| Release gate | Test suite: python -m pytest |
| Support handoff | Name the owner, escalation path, rollback path, known limits, and review cadence before a paid or production pilot. |

## Verification Surface

| Purpose | Command |
|---|---|
| Test suite | `python -m pytest` |

## CI Surface

- .github/workflows/architecture-blueprint.yml
- .github/workflows/ci.yml
- .github/workflows/dependency-review.yml
- .github/workflows/export-onnx.yml
- .github/workflows/repository-health.yml
- .github/workflows/repository-surface.yml
- .github/workflows/secret-scan.yml

## Acceptance Criteria

- python -m pytest can be run or the equivalent CI gate is visible.
- README, review guide, quality notes, service model, and this readiness note agree on the same scope.
- Demo, fixture, synthetic, or public-data boundaries are explicit before a buyer sees outputs.
- A reviewer can identify the first useful outcome without reading implementation details.
- Production claims stay behind customer-specific validation, access control, monitoring, and support handoff.

## Integration Path

- Run a synthetic-data walkthrough with the buyer and document the acceptance criteria.
- Scope a controlled pilot using approved data, named users, secrets, and rollback paths.
- Convert the pilot into an operating handoff with monitoring, review cadence, support owner, and renewal metric.

## Proof Points

- pytest passes
- Model card exists
- API detection path is documented

## Operating Metrics

- mAP/recall by defect
- False-negative review
- Operator review time

## Open Risks

- Human inspection required
- Site-specific validation
- No production acceptance without criteria

## Finish Line

- Keep the public repository honest, runnable, and easy to review.
- Keep sensitive data, secrets, private tenant details, and unsupported claims out of public artifacts.
- Treat this repository as a proof surface until an approved pilot defines users, data, access, monitoring, support, and success metrics.
