# ADA External Knowledge

External content is ingested as untrusted, provenance-linked documents. Each source has an ID, URI, publisher, retrieval timestamp, optional content digest, license ID and terms URI. New content is PENDING; only explicit verification can mark it VERIFIED, REJECTED or QUARANTINED. Only verified documents may be chunked or retrieved.

The initial reference implementation provides immutable contracts, digest checking, collision detection, explicit license allowlisting, deterministic text normalization, overlap chunking and a simple lexical retrieval baseline. It performs no network requests, browser automation, OCR or semantic extraction.

The existing ADA Memory/Experience subsystem similarly separates candidate and validated states. This subsystem applies the same distinction at the external-data boundary. The [NIST GenAI Profile](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence) highlights governance, content provenance and pre-deployment testing as risk-management areas.

Limitations: in-memory bounded storage only; license allowlisting is a policy hook, not legal advice or complete rights clearance; lexical ranking is not semantic retrieval or source verification; crawlers and browser adapters must be separately permissioned.
