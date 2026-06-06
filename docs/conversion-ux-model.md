# Conversion UX Model - Weld Defect Vision

Updated: 2026-05-30

This note specializes the repository for service launch. It combines product strategy, UX design, behavioral economics, and neuroscience-informed attention and working-memory design in a practical way: reduce confusion, build trust, help the right user act, and avoid manipulative conversion patterns.

## Commercial Focus

| Field | Decision |
|---|---|
| Repository status | active |
| Lane | B2B industrial AI validation |
| Primary buyer or user | Manufacturing quality teams, welding inspection groups, industrial AI, and edge deployment reviewers. |
| Value wedge | YOLOv8 defect workflow with model governance, serving, and operator-readable evidence. |
| Service model | Paid diagnostic, fixed-scope pilot, and retained operating review |
| Operating note | Start with a small risk-reversing review, then convert to a controlled pilot with success metrics. |
| Best channel | Founder-led outreach, one-page scorecards, recorded demos, and domain-specific checklists. |

## UX Positioning

| Moment | Design decision |
|---|---|
| First screen | State the buyer, painful workflow, proof artifact, and next action in one compact view. |
| First action | Open the review guide, run or inspect pytest passes, and map one buyer workflow to the pilot checklist. |
| Proof moment | Show a generated artifact, benchmark, report, replay, export, or review pack before any paid ask. |
| Trust moment | Put boundaries, data policy, unsupported claims, and human-review points beside the result. |
| Conversion moment | Offer the smallest next step that matches the user's risk level. |
| Retention moment | Bring the user back with saved evidence, scorecards, review cadence, templates, or repeatable workflows. |

## Behavioral Design

| Principle | Application |
|---|---|
| Attention and working memory | Use one primary action, one visible proof artifact, and one next step so the interface does not overload attention. |
| Cognitive fluency | The first screen should answer who it is for, what pain it removes, what proof exists, and what action comes next. |
| Chunking | Break the path into inspect, try, trust, decide. Avoid making the buyer hold the whole system in working memory. |
| Salience | Show one concrete pain metric or before/after artifact instead of a broad value claim. |
| Trust calibration | State boundaries, unsupported claims, data limits, and human-review points before conversion prompts. |
| Choice architecture | Offer three clean next steps: inspect proof, run demo/check, or discuss a scoped pilot. |
| Loss aversion, used carefully | Show operational waste, review delay, or audit exposure with evidence; do not use fear without proof. |
| Authority through evidence | Use CI, evals, runbooks, fixtures, and exported artifacts as proof instead of borrowed prestige. |
| Goal-gradient effect | Show pilot progress as steps completed toward an operating handoff. |

## Design System Direction

- Use dense but calm dashboards: tables, status chips, timelines, evidence panels, and clear severity hierarchy.
- Show source, decision, owner, boundary, and next action together so the reviewer never hunts for trust context.
- Use restrained color: neutral base, semantic status colors, no decorative gradients where operators need clarity.

## Conversion Path

- Risk-reversing entry: Inspection PoC (scope after buyer intake) with one acceptance metric.
- Pilot: Model validation study (scope after buyer intake) using buyer-approved data and named operators.
- Recurring layer: Edge-serving readiness assessment (scope after buyer intake) for monitoring, governance, support, or managed review.

## Scope Frame

- Anchor scope to the buyer's existing cost: hours lost, incidents, review delay, audit exposure, or manual handoff.
- Use the first offer as risk reversal, not as a race to the bottom.
- Put mAP/recall by defect on the pilot scorecard.

## Metrics To Watch

- mAP/recall by defect
- False-negative review
- Operator review time

## Ethical Guardrails

- No fake users, fake logos, fake financial outcomes, fake benchmarks, or unverifiable endorsements.
- No urgency timers, hidden opt-outs, forced continuity, or confusing scope.
- Conversion prompts should come after value or evidence, not before.
- Data collection should be minimal, visible, and tied to product value.
- Human inspection required
- Site-specific validation
- No production acceptance without criteria

## Next UI/UX Upgrade

- Add one above-the-fold path that leads to the first proof action.
- Add one trust panel beside the proof output, not hidden in legal text.
- Add one buyer-specific next step: diagnostic, workshop, pilot, package, support, or revival checklist.
- Remove any copy that asks for belief before showing evidence.
