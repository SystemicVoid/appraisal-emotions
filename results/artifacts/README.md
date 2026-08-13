# Fitted-direction artifacts (drop-in slot)

This directory is gitignored except for this README. Place here either:

- `reveal_directions.json` + `reveal_directions.directions.npz` copied from the parent
  program's certified run (see `../ra_prime_certification.md` for the digest to verify), or
- the same pair produced locally by `just extract-rpe`, plus
- `emotion_vectors.json` + `.npz` produced by the E0 tier.

Analysis commands (`map-geometry`, `expectation-control`) read from this directory by
default. Artifacts are hash-bound: loaders verify the npz payload digests recorded in the
JSON before use.
