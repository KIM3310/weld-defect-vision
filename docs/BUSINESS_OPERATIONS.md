# Business Operations Readiness

This document turns the demo/repository into an operating business checklist without performing irreversible actions. It was generated on 2026-06-26 from the portfolio audit and current public docs. It is operational planning, not legal, tax, or financial advice.

## Commercial position

- Priority: **P1-proof-asset**
- Monetization path: **portfolio proof asset**
- Secondary path: **public demo/readme polish candidate**
- Readiness score: **80/100**
- Visibility: **public**
- Archived: **false**

## Deployment lane: static-preview-first

- Use a preview deployment first; Cloudflare Pages or GitHub Pages are the lowest-friction static lanes.
- Before custom domain/DNS, verify no secrets, no customer data, and no unsupported revenue claims are published.
- Prefer screenshots or a gated preview link for sales until privacy/support/payment pages are approved.

## Payment lane: manual-invoice-first

- Use manual invoice/proposal until offer, support, refund, and privacy boundaries are approved.
- Hosted payment links can be added later without code once account setup is complete.

## Privacy and data lane: privacy-standard-minimize-data

- Inventory personal data, customer data, logs, analytics identifiers, uploaded files, and model prompts before launch.
- Collect the minimum data needed; define retention, deletion, access control, incident response, and data export/deletion request handling.
- Publish a plain-language privacy policy before collecting contact, analytics, payment, or uploaded-file data; this draft is not legal advice.

## Customer support lane: best-effort-community-support

- Use GitHub Issues/Discussions for non-sensitive feedback.
- Do not promise response times until a paid plan or pilot contract exists.

## Launch blockers that must stay explicit

- Payment/KYC/tax setup requires account-owner action; no payment link is live from this automation.
- Privacy policy/terms/refund language requires owner/legal review before customer data or money collection.
- Production launch, custom domain/DNS, analytics, and support inbox changes require explicit approval.

## Pre-launch checklist

- [ ] Repo-specific verification passes locally and/or in CI.
- [ ] Secret-pattern audit findings are reviewed and resolved or documented as false positives.
- [ ] Public copy avoids revenue guarantees and unsupported legal/medical/financial/security claims.
- [ ] Privacy policy, terms/refund policy, and support scope are approved by the owner before publication.
- [ ] Payment account/KYC/tax configuration is complete before accepting money.
- [ ] Support inbox, escalation owner, response window, and customer-data handling are ready.
- [ ] Production deployment, custom domain/DNS, analytics, and email capture are explicitly approved.

## Support macros

- **Bug intake:** ask for environment, reproduction steps, expected/actual result, logs with secrets removed, and impact level.
- **Paid pilot intake:** capture buyer, use case, data sensitivity, success metric, deadline, access constraints, and decision owner.
- **Refund/escalation:** acknowledge within the promised support window, preserve the order/customer reference privately, and escalate policy exceptions to the owner.
- **Data request:** verify requester identity through the approved support channel, avoid public issue threads, and log deletion/export actions.

## Sources checked

- Stripe Payment Links: https://docs.stripe.com/payment-links
- Stripe create payment link: https://docs.stripe.com/payment-links/create
- Cloudflare Pages docs: https://developers.cloudflare.com/pages/
- Cloudflare Pages Direct Upload: https://developers.cloudflare.com/pages/get-started/direct-upload/
- GitHub Pages publishing source: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
- FTC Privacy and Security: https://www.ftc.gov/business-guidance/privacy-security
- FTC Protecting Personal Information: https://www.ftc.gov/business-guidance/resources/protecting-personal-information-guide-business
