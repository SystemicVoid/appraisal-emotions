# appraisal-emotions

Code and results for studying whether emotion-concept representations in
Qwen3.6-27B track reward prediction error, the difference between a gamble's realised
reward and its stated expected value.

We extract appraisal directions from activations on affect-neutral gambles and
emotion-concept vectors from separately generated stories. We then compare their
geometry, vary reward and expectation independently, and test whether interventions
on the RPE representation affect later choices.

Read the [paper](submission.md) or [PDF](submission.pdf) for methods and results.

## Findings

- Emotion-concept readouts track reward relative to expectation. The pattern replicates
  with an alternative story-generation recipe.
- Prior outcome and expectation affect subsequent risk-taking, but the behavioural
  experiment does not distinguish which variable carries over.
- The interventions do not establish that the identified RPE representation causes
  the behavioural effect. The geometry results beyond affect-concept valence remain
  suggestive.

These findings concern representations and behaviour, not subjective experience.

## Run locally

Requires Python 3.12 or later, `uv`, and `just`.

```bash
uv sync
just test

# Run the extraction and analysis pipeline on a fake backend, without a GPU.
just extract-rpe-smoke
just extract-emotions-smoke
just map-geometry
just expectation-control
just patch-reveals
```

The fake backend checks the pipeline; its outputs are not research results.
For real-model setup and GPU runs, see the [Lambda runbook](docs/agents/lambda-runbook.md).
The [justfile](justfile) lists commands and their arguments.

## Repository

- [runs/](runs/) contains recorded experiment reports and artifacts.
- [configs/](configs/) contains experiment settings and pinned model revisions.
- [src/appraisal_emotions/](src/appraisal_emotions/) contains stimulus generation,
  activation capture, and analysis code.
- [docs/design/](docs/design/) contains experiment designs and follow-up analyses.
- [docs/literature.md](docs/literature.md) lists related research and source caveats.

By Hugo Nguyen and Artyom Chelbayev, with Apart Research at the Digital Minds Research
Sprint, August 2026. The RPE extraction pipeline builds on prior unpublished work
recorded in [the provenance notes](results/ra_prime_certification.md).
