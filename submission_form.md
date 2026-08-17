# Submission form — copy/paste draft

Working sheet for Hugo and Artyom. Updated 2026-08-17 against the `paper-final` branch.

Each field is tagged:

* **SETTLED** — copy straight into the form, already matches `paper-final/submission.md`.
* **DECIDE** — a choice one of us still has to make.
* **NEEDED** — a detail only you two have; I could not source it from the repo.

> **Blocker before submitting.** The abstract on `paper-final` is currently **195 words**, over the
> sprint's 150-word limit. The Project Summary below quotes it verbatim, so the paper and this form
> have to be trimmed together. A pre-trimmed 147-word version is provided under the summary field.

## Project Details

**Project Title\*** — SETTLED

> An LLM's Emotion-Concept Readout Tracks Reward Prediction Error

**Project Summary\*** — SETTLED, pending the trim above

Current abstract, verbatim from `paper-final` (195 words — over limit):

> Reward prediction error (RPE) is the difference between a received reward and a predicted one, a computation first linked to dopaminergic activity in non-human primates (Schultz, Dayan & Montague, 1997). In humans, RPE also predicts momentary subjective well-being: outcomes that beat expectations are associated with greater happiness than equivalent outcomes that were already expected (Rutledge et al., 2014). We test whether a language model performs an analogous computation, and whether emotion-concept representations built independently of that task reflect it. In Qwen/Qwen3.6-27B, signed RPE is strongly separable in the residual stream on affect-neutral gambles (AUROC 0.985, against a 0.734 random-direction floor). Two emotion-concept readouts, derived from generated stories rather than from the gambles, track reward relative to expectation rather than either quantity on its own: holding reward fixed and holding expected value fixed both produce shifts, and the two are comparable in size (ratios 1.11 and 1.25; all permutation p = 1/10001). In a sequential gambling task, positive versus negative prior outcomes are associated with a +0.19-logit shift toward subsequent risk-taking (p ≈ 10^-4; 66% of 209 matched pairs). Our intervention experiments do not yet establish that the identified RPE representation causally mediates this behavioural effect.

Optional pre-trimmed version (**147 words**, verified) if you want the paper and form to land
under the limit without re-drafting. Same claims, same numbers; it drops the random-direction
floor, the restatement of the two matched shifts, and the 66%-of-pairs detail, all of which
survive in the body text and appendix:

> Reward prediction error (RPE) is the gap between received and predicted reward, first linked to dopaminergic activity in non-human primates (Schultz, Dayan & Montague, 1997). In humans, RPE also predicts momentary subjective well-being: outcomes that beat expectations bring more happiness than equivalent expected ones (Rutledge et al., 2014). We test whether a language model performs an analogous computation that independently built emotion-concept representations reflect. In Qwen/Qwen3.6-27B, signed RPE is strongly separable in the residual stream on affect-neutral gambles (AUROC 0.985). Two emotion-concept readouts, derived from generated stories, not the gambles, track reward relative to expectation, not either quantity alone (ratios 1.11 and 1.25; all permutation p = 1/10001). In a sequential gambling task, positive versus negative prior outcomes are associated with a +0.19-logit shift toward subsequent risk-taking (p ≈ 10^-4). Our intervention experiments do not yet establish that the identified RPE representation causally mediates this behavioural effect.

**Upload your PDF report\*** — ACTION

> Export `submission.md` from the `paper-final` branch. It pulls in Figures 1–4 from `figures/`.
> Do this after the abstract trim so the PDF and the form agree.

**Are you interested in publishing this project?\*** — SETTLED

> Yes

**Pick one or more tracks\*** — NEEDED

> The four track names are not recorded anywhere in the repo, and the submission form only shows
> them as "Track 1–4". The project sits in the Digital Minds Research Sprint (per the paper byline
> and the README). Someone with the portal open needs to pick.

## Optional Uploads

**Presentation Recording** — DECIDE

> Add a link if one gets recorded; otherwise leave blank.

**Project Code** — SETTLED

> https://github.com/SystemicVoid/appraisal-emotions

**Upload your slideshow** — DECIDE

> Optional; nothing in the repo yet.

**Upload your project image** — DECIDE

> `figures/fig1_matched_effects.png` is the most self-explanatory single image — it carries the
> central result. `figures/fig4_depth_profile.png` is the most visually striking.

**Additional Material** — SUGGESTED

> * `docs/design/submission-factcheck.md` — the artifact-by-artifact fact-check of every number in
>   the report. This is unusually strong evidence of rigour and is worth surfacing.
> * `docs/literature.md` — annotated bibliography with verification caveats.
> * `docs/design/experiment.md` — full pre-registered design.

## Team Details

**Team Name\*** — NEEDED

> Not recorded anywhere in the repo.

**Location\*** — NEEDED

> The form asks which city you are joining from. If you are in different cities, check whether the
> portal wants one location per team or per member.

**Team Member Name\*** — SETTLED

> Hugo Nguyen (affiliation: Independent)
> Artyom Chelbayev (affiliation: Independent)
>
> Affiliations are taken from the author line on `paper-final`. If either of you wants a real
> affiliation instead of "Independent", it has to change in both places.

**Team Member Email\*** — PARTLY NEEDED

> Artyom: a.chelbayev@gmail.com — please confirm this is the address you want on the submission.
> Hugo: NEEDED.

**Team Member Discord Username** — NEEDED

> Optional on the form, but the sprint usually coordinates over Discord, so worth filling in.

**Team Member Google Scholar Link** — DECIDE

> Optional. Skip unless either of you wants the citation trail.

**Do you need to add more team members?** — SETTLED

> Yes — two members: Hugo Nguyen and Artyom Chelbayev.

## Open items on the paper itself

Tracked here so they do not get lost between the paper and the form:

1. Abstract trim to ≤150 words (see blocker above).
2. The `Author Contributions` section was removed from `submission.md`. If the sprint template
   requires one, it needs restoring before export.
3. The LLM Usage Statement still carries a `TODO (team): confirm wording before submission`.
