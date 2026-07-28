# Search Growth Implementation - Weld Defect Vision

This repository now exposes a search-readable service surface in addition to the system architecture. The implementation is designed to support organic discovery, AI answer surfaces, and a free-to-paid service path without committing to paid infrastructure first.

## Implemented Surface

| Surface | Path |
| --- | --- |
| Machine-readable offer | [docs/service-offer.json](./service-offer.json) |
| Revenue architecture | [docs/revenue-architecture.md](./revenue-architecture.md) |
| System architecture | [docs/system-architecture.md](./system-architecture.md) |
| Public canonical URL | https://kim3310.github.io/weld-defect-vision/ |
| Lead capture URL | https://kim3310-doeon-kim-portfolio.pages.dev/?offer=weld-defect-vision&inquiry=industrial-validation-discovery#private-inquiry |
| Commercial route | https://kim3310-doeon-kim-portfolio.pages.dev/?offer=weld-defect-vision#service-offers |

## Search Positioning

- Primary query: Weld Defect Vision industrial validation discovery
- Secondary queries: Weld Defect Vision demo; Weld Defect Vision system architecture; Weld Defect Vision synthetic inspection demo; weld defect data suitability validation discovery
- Public entry point: free synthetic-data inspection demo and architecture page
- Paid boundary: private industrial validation discovery for data suitability, baseline evaluation, model-card draft, and human-review acceptance criteria

## Conversion Boundary

The public surface stays crawlable, synthetic, and conservative. Private value starts when a visitor wants data suitability review, baseline evaluation planning, model-card drafting, and human-review acceptance criteria.

## Deployment Notes

- Keep the sitemap and robots file aligned with the final production domain.
- Submit the canonical URL and sitemap in Google Search Console after the domain is connected.
- The lead-capture path is the central private inquiry URL for the `industrial-validation-discovery` lane.
- Keep exact free-tier quotas out of public promises because provider limits change.
