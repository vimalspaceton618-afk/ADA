# Initialization Health Checks

**Status:** Proposed acceptance probes

## Probe classes

### Configuration
- parse and schema validation;
- cross-field invariants;
- canonical digest generation;
- unknown-key rejection except documented extension namespaces.

### Software
- Python and framework version;
- dependency lockfile consistency;
- tokenizer package/API availability;
- optional kernel package import and capability probe.

### Hardware
- device visibility and identity;
- small allocation and synchronization;
- dtype operation probe;
- backend-specific kernel probe when required;
- host/device memory snapshot.

### Data and artifacts
- manifest readability and digest;
- tokenizer revision and vocabulary match;
- checkpoint digest and metadata compatibility;
- output directory writable and disk space sufficient.

### Model compatibility
- implementation declares every requested component;
- tensor dimensions and head-group divisibility;
- causal mask tests;
- variable-length and padding tests;
- reference-equivalence tests where a reference exists.

## Health states

| State | Meaning | Action |
|---|---|---|
| PASS | required probe succeeded | continue |
| WARN | optional feature unavailable | continue only if profile permits |
| FAIL | required capability absent or invalid | reject startup |
| SKIP | not applicable | record rationale |

Final status cannot be PASS if any required probe is FAIL or an unapproved SKIP.

## Diagnostic record

Every probe records ID, stage, start/end time, result, duration, resource consumption where relevant, exception code, remediation and run ID. Secrets/personal data are redacted.

## Notebook-derived test priorities

- scan equivalence against sequential reference;
- non-power-of-two sequence lengths;
- gradient equivalence and finiteness;
- causal invariance to future-token perturbations;
- MoE top-k and routing behavior;
- memory shape and finite gradients;
- tokenizer vocabulary vs embedding/head dimensions;
- checkpoint architecture compatibility.

Small component tests do not validate full-model training, hardware scaling or model quality.
