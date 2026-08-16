# Appraisal Geometry: Reward Prediction Error and Emotion-Concept Representations in Qwen3.6-27B

<!-- TITLE: placeholder — pick final title. Alternatives:
     "Does an LLM's Emotion-Concept Readout Track Its Reward Prediction Error?"
     "Reward Prediction Error as an Internal State Variable in an LLM" -->

**Authors:** [Hugo Nguyen — affiliation], [Artyom Chelbayev — affiliation]
*With Apart Research. Research conducted at the Digital Minds Research Sprint, August 2026.*

## Abstract

Reward prediction error (RPE) theory holds that reward signals reflect the difference between received and predicted rewards, a computational account originally linked to dopaminergic activity in non-human primates (Schultz, Dayan & Montague, 1997). In humans, reward prediction errors have also been shown to predict momentary subjective well-being, such that outcomes that exceed expectations are associated with greater happiness than equivalent outcomes that were already expected (Rutledge et al., 2014). We apply RPE theory to Qwen/Qwen3.6-27B, combining activation probing on affect-neutral gambles with independently constructed emotion-concept readouts and a sequential risk-choice task. We find that signed RPE is strongly separable in the model's residual stream (AUROC 0.985), and that the model's emotion-concept readout tracks the reward-versus-expectation comparison itself rather than either component independently (effect ratios 1.11 and 1.25; all permutation p = 1/10001). We further find that positive versus negative prior outcomes are associated with a +0.19-logit shift toward subsequent risk-taking (p ≈ 10^-4), although our intervention experiments do not yet establish that the identified RPE representation causally drives this behavioural effect.

## 1. Introduction

Considering a large language model (LLM) as a digital mind presents ample avenues of human- and animal-inspired research, amongst which are the questions of well-being, its source, and influence on subsequent actions. As LLMs increasingly take actions in the world through agentic systems, understanding the underlying mechanisms of those actions can yield insights into pre-existing biases and give ways of counteracting them. If a model's recent outcome history shifts its subsequent risk preferences — as prior outcomes shift human choices — then recent outcome/expectation history is an internal state variable worth monitoring in deployed agents.

In this paper we set out to test the following hypotheses:

* emotion-concept readout at the outcome token tracks the model's reward-prediction error;
* the RPE representation computed at the outcome reveal is functionally used in the model's subsequent risk-taking choice.

Our main contributions are:

1. We certify a stable, separable representation of signed reward prediction error in the residual stream of Qwen3.6-27B on affect-neutral gambles (AUROC 0.985 against a 0.734 random-direction floor), extending a prior certification of the same computation in a 4B-parameter model.
2. We show that emotion-concept readouts constructed entirely independently of the gambling task track the reward-versus-expectation *comparison* — the same `a ≈ −b` coefficient signature through which Rutledge et al. (2014) identify the RPE contribution to human momentary well-being — rather than reward or expectation alone.
3. We find a behavioural carry-over effect consistent with the house-money effect (Thaler & Johnson, 1990): positive prior outcomes shift the model toward riskier subsequent choices by +0.19 logits. We then report an intervention programme that is deliberately honest about what it does and does not establish, including a passthrough decomposition that exposes how same-position activation patching can mimic functional transfer through the residual stream's identity path.
4. As a pilot-suggestive robustness result, we find that outcome-linked positive emotion concepts carry excess alignment with the RPE direction beyond what human-rated valence and arousal norms predict.

## 2. Related Work

Schultz, Dayan & Montague (1997) established reward prediction error as the computational account of dopaminergic reward signalling, and Rutledge et al. (2014) showed that in humans momentary subjective well-being tracks recent RPEs rather than reward levels — their computational model identifies the RPE contribution through matched reward and expectation coefficients of opposite sign. Our readout analysis transplants exactly this identification strategy to a language model's internal activations.

Our work builds directly on currently unpublished research by Hugo Nguyen indicating that Qwen/Qwen3-4B-Instruct-2507 performs RPE computations consistent with the Rutledge et al. (2014) model. <!-- FLAG (Hugo): the earlier draft said "unpublished peer-reviewed research", which is self-contradictory — please set the correct status (under review / working paper / etc.) -->

For emotion-concept representations, we rely on the methodology of Sofroniew et al. (2026), who construct emotion-concept vectors from model activations over generated stories and study their function in a large language model. We adopt their extraction recipe but construct our own appraisal-structured word list, organised around whether a concept semantically requires an outcome-versus-expectation comparison — the specific question their general-purpose word list does not address. Berg (2026) argues on computational grounds that emotion concepts should be organised around goal-relative prediction error; our measurements offer representational-level evidence bearing on that premise.

For the behavioural experiment, Thaler & Johnson (1990) document increased risk-seeking after prior gains (the house-money effect) and after losses when a bet offers a chance to break even, while Isen & Patrick (1983) report a stake-dependent interaction between induced positive affect and risk-taking. Together these motivate testing whether a model's revealed outcome carries over into its next choice, without pre-committing to a sign.

## 3. Methods

**RPE certification.** First, we test whether RPE computations scale from a previously researched smaller Qwen/Qwen3-4B-Instruct-2507 model to the larger Qwen/Qwen3.6-27B, which is the base of subsequent experiments. Throughout this paper, RPE refers to described-EV RPE: realised reward minus the explicitly stated expected value (EV) of the gamble. The test battery consists of 1,984 deliberately neutral scenarios describing a 50/50 gamble between two outcomes, followed by a reveal of which outcome actually occurred. The model's internal activations are recorded at the outcome token, to test whether a clean, separable RPE signal exists in its internal representations. Outcome labels are meaningless three-letter codes and point values are round numbers, with the framing sentence rotated across several neutral phrasings, so the design contains no emotional language that could confound the results. Directions are fitted on 1,488 estimation trials and model depth selected separately on 496 held-out selection trials.

**Emotion-concept map.** Second, we construct an emotion-concept map following Sofroniew et al. (2026). For the primary emotion-concept readout, we use 84 emotion words organised into appraisal-structured families, including outcome-linked positive and negative concepts and matched non-outcome controls. For example, `disappointed` specifically implies an outcome that falls short of an expectation, whereas `sad` does not necessarily require such an expectation-outcome comparison. For each word, we generate stories in which a character experiences a situation associated with that concept without naming the target emotion. We present these stories to the model and average their residual-stream activations to construct one emotion-concept vector per word, centred against the mean across all words. From these independently constructed vectors, we derive two readout axes used in our subsequent RPE experiment: the first principal component of the emotion-concept space, validated against human-rated valence (Mohammad, 2025), and the pre-specified `elated − disappointed` direction. We later widen the map to 111 words and 24 stories per word for robustness and geometry analyses, where we additionally residualize on these same human-rated valence and arousal norms to test whether alignment with RPE exceeds what they alone would predict.

**Expectation control.** Third, we test whether the model's emotion-concept readout tracks reward prediction error rather than reward or expectation independently. We project reveal-token activations onto the two independently constructed emotion-concept axes. We then exploit the matched structure of the reveal battery. In reward-matched cells, realised reward is held constant while stated expected value varies; in EV-matched pairs, expected value is held constant while realised reward varies. We model the readout as `projection ≈ a·reward + b·EV`. A reward-prediction-error readout predicts contributions of opposite sign and comparable magnitude (`a ≈ −b`), whereas a pure outcome tracker or pure expectation tracker predicts one of the two matched effects to disappear. Before analysis, we specify a factor-of-two tolerance for the relative coefficient magnitudes and evaluate effects using 10,000 permutations.

**Behavioural carry-over.** Fourth, we test whether information from a recently revealed outcome carries over into a subsequent action. We present the model with two consecutive gambles: the first produces a positive or negative RPE, while the second asks the model to choose between a certain outcome and a risky gamble. We vary the certain reward across three levels around a risky option with expected value 20, allowing us to place the model's choice close to indifference and measure whether its preference for risk differs following positive versus negative first-round outcomes.

**Interventions.** Fifth, we test the stronger causal hypothesis by intervening directly on the model's internal representation. We first patch the RPE-related component of the residual stream at the outcome reveal and measure its downstream emotion-concept readout. Because a residual stream can mechanically carry an injected vector forward, we compare the measured transfer against the amount predicted by passive passthrough and against no-op and random-direction controls. We then use a stronger cross-position design in which the intervention is written at the first-round outcome token while behaviour is measured several tokens later at the second-round choice. A pre-specified power gate determines whether the behavioural intervention is sufficiently sensitive to interpret, while a reachability control tests whether an intervention at the outcome token can affect the later choice position at all.

Our code is publicly available at https://github.com/SystemicVoid/appraisal-emotions.

## 4. Results

**Emotion-concept readout at the outcome token tracks the model's reward-prediction error.** We first verified that Qwen3.6-27B contains a stable, separable representation of signed RPE at the outcome reveal (AUROC 0.985; full certification in Appendix A1). We then tested whether the model's independently constructed emotion-concept readout tracked this comparison rather than reward or expectation alone. Holding reward constant while varying expectation, and separately holding expectation constant while varying reward, shifted both the general emotion-concept valence axis and the pre-specified `elated − disappointed` axis in the directions predicted by `reward − expectation`. The two components were comparable in magnitude (ratios 1.11 and 1.25; all four permutation p = 1/10001), supporting the hypothesis that the model's emotion-concept readout tracks the computed RPE itself rather than either of its constituent variables independently. This `a ≈ −b` coefficient structure is the same functional signature through which Rutledge et al. (2014) identify the RPE contribution to momentary subjective well-being in humans.

> **[Figure 1 here]** *Matched-design coefficient structure: reward-matched and EV-matched effects on both emotion-concept axes, with the a ≈ −b prediction. (figures/fig1_matched_effects.*)*

**The prior outcome/expectation manipulation carries over into the model's subsequent risk-taking choice; causal use of the RPE representation remains open.** On unpatched trials, after a positive-RPE reveal the model prefers the risky option on the subsequent gamble by +0.19 logits more than after a negative-RPE reveal (p ≈ 10^-4; 66% of 209 matched pairs positive). The direction is consistent with the house-money effect described by Thaler & Johnson (1990), in which favourable prior outcomes increase subsequent risk-taking. However, within this behavioural design expected value and signed RPE are not independently identified, so the result establishes that the prior outcome/expectation manipulation carries over into later choice rather than that signed RPE specifically causes the change. Our activation interventions likewise do not yet establish that the identified RPE representation is causally used to produce the subsequent choice.

> **[Figure 2 here]** *Behavioural carry-over: per-pair risk-choice logit gap after positive versus negative first-round outcomes. (figures/fig2_carryover.*)*

### 4.1 Controls and robustness

For our first hypothesis, we progressively test whether the relationship between RPE and the emotion-concept readout can be explained by simpler alternatives. The gamble surface itself contains no emotion words or valence-bearing outcome labels, reducing the possibility that the readout reflects lexical leakage. More importantly, the reward-matched and EV-matched designs separate the two components of RPE: a pure outcome tracker predicts no effect when reward is held constant, while a pure expectation tracker predicts no effect when expected value is held constant. Both alternatives are inconsistent with the observed effects on both emotion-concept axes.

We then ask whether the relationship extends beyond the broad positive–negative structure of the emotion-concept map. In a widened analysis using 111 words and 24 stories per word, outcome-linked positive concepts showed +0.0346 excess alignment with RPE relative to non-outcome positive controls after accounting for human-rated valence and arousal (permutation p = 0.0177), clearing both label-shuffle and random-direction floors. This provides convergent evidence for our first hypothesis, but we treat it as pilot-suggestive: the effect remains below the experiment's 80% minimum detectable effect, and generated outcome-linked stories differed systematically in event structure from their controls. Detailed geometry results and sensitivity analyses are reported in Appendix A2.

> **[Figure 3 here]** *Word-level RPE alignment residuals by family (widened run, block 35), showing the outcome-linked positive family's excess over non-outcome positive controls. (figures/fig3_family_residuals.*)*

For our second hypothesis, the robustness analysis distinguishes behavioural carry-over from causal use of the identified RPE representation. Our first intervention appeared to support a causal interpretation: patching the RPE-related component at the outcome token moved the downstream emotion-concept readout. However, a passthrough decomposition showed that 79.5–84.0% of the apparent PC1 transfer and 97–105% of the `elated − disappointed` transfer was predicted simply by the injected vector remaining in the residual stream, with no direction-specific excess over the no-op control. We therefore interpret this experiment as showing that the signal is carried, not that it is functionally used.

We subsequently removed this direct identity path by patching at the first-round outcome token and measuring behaviour at the later second-round choice position. In the widened run, a reachability control passed: a full-residual intervention at the outcome token shifted later answer-slot logits by −0.196 logits (p = 3 × 10^-4), establishing that interventions at the reveal token can affect downstream computation (the smaller base run had not passed this control). However, the behavioural power gate failed in both runs. Even after widening the experiment, the minimum detectable effect remained 0.130 against a pre-specified target of approximately 0.095. The causal claim therefore remains unresolved: the experiment establishes behavioural carry-over and downstream reachability, but is not sufficiently sensitive to determine whether the identified RPE representation itself causes the subsequent choice.

## 5. Discussion and Limitations

Our results bear on how agentic AI systems should be monitored. If recent outcome/expectation history is carried in a model's internal state and co-varies with its subsequent risk-taking, then an agent's recent "wins" and "losses" are a candidate internal state variable for interpretability-based oversight — independent of any claim about experience. The convergence between the readout signature identified here and the human RPE-to-well-being signature of Rutledge et al. (2014) also suggests that appraisal-theoretic designs — varying expectation and outcome independently — transfer productively to language-model interpretability.

**Limitations.** Our strongest claims are measurement-validity claims: a direction exists, is separable, and its readout tracks a specific comparison. (1) Causal use of the RPE representation in behaviour is not established; the cross-position intervention's power gate failed in both runs, so its null patched-behaviour results are uninterpretable rather than negative. (2) The behavioural carry-over design cannot distinguish signed RPE from expected value as the carried variable. (3) The geometry result is a pilot-suggestive checkpoint, not a powered confirmation: planned power was 0.36, and the observed effect is below the realised minimum detectable effect. Its floor clearance is also fragile to removal of a single control word. (4) Word-level predictions failed in diagnosable ways: numeric valence norms mis-rate several words in the intended sense (e.g. `resigned`), and generated stories differed in event structure between semantic families, so word-level comparisons partly reflect instrument misspecification. (5) All results come from one model family and one operationalization of RPE (described-EV at a revealed gamble outcome); reward levels, probabilities, and framing were deliberately narrow. (6) The prior 4B-model certification we build on is unpublished; its headline number is inherited from that work, not reproduced in this repository.

**Future work.** A re-powered cross-position intervention could settle whether the RPE representation is causally used in later action selection. Activation patching of emotion-concept vectors themselves would probe what functional role those representations play. A powered replication of the geometry checkpoint should pre-specify the block-48–50 depth band, use sense-checked norms with a binary-valence arm, and event-audit its stimuli. Reproduction across model families and sizes would test transferability.

## 6. Conclusion

Our findings indicate that RPE is represented in an LLM, that the model's emotion-concept readout tracks this computation, and that related outcome/expectation information co-varies with subsequent action. In Qwen3.6-27B, the signed-RPE representation is stable (split-half stability 0.925) and strongly decodable (AUROC 0.985), consistent with the prior certification of the same computation in Qwen3-4B-Instruct-2507 (split-half stability 0.911 in that prior work). The model's emotion-concept readout tracks the RPE computation, with reward and expectation effects closely matched on both tested axes (ratios 1.11 and 1.25; all permutation p = 1/10001). Finally, an LLM's subsequent action co-varies with the most recent outcome/expectation manipulation: positive versus negative prior outcomes were associated with a +0.19-logit shift toward risk-taking (p ≈ 10^-4), although our intervention experiments do not yet establish that the identified RPE representation causally drives this effect. As such, recent outcome/expectation history may be a relevant internal state variable to understand and monitor within agentic systems.

Finally, nothing in our research was aimed at, implies or supports phenomenal experience of emotion concepts by LLMs. Whether these representations have any connection to phenomenal experience or AI welfare remains an open empirical question.

## Code and Data

* Code repository: https://github.com/SystemicVoid/appraisal-emotions
* Data: run artifacts (JSON reports) are versioned in the repository; full activation tensors and logs are attached to repository releases.

## Author Contributions

[TODO — e.g. "H.N. designed the experiments and led the analysis. A.C. ... Both authors contributed to writing and reviewed the final manuscript."]

## References

1. Schultz, W., Dayan, P., & Montague, P. R. (1997). "A Neural Substrate of Prediction and Reward." *Science, 275*(5306), 1593–1599. https://doi.org/10.1126/science.275.5306.1593
2. Rutledge, R. B., Skandali, N., Dayan, P., & Dolan, R. J. (2014). "A Computational and Neural Model of Momentary Subjective Well-Being." *Proceedings of the National Academy of Sciences, 111*(33), 12252–12257. https://doi.org/10.1073/pnas.1407535111
3. Sofroniew et al. (2026). "Emotion Concepts and their Function in a Large Language Model." *arXiv:2604.07729.*
4. Mohammad, S. M. (2025). "NRC VAD Lexicon v2: Norms for Valence, Arousal, and Dominance for over 55k English Terms." arXiv:2503.23547.
5. Thaler, R. H., & Johnson, E. J. (1990). "Gambling with the House Money and Trying to Break Even: The Effects of Prior Outcomes on Risky Choice." *Management Science, 36*(6), 643–660. https://doi.org/10.1287/mnsc.36.6.643
6. Isen, A. M., & Patrick, R. (1983). "The Effect of Positive Feelings on Risk Taking: When the Chips Are Down." *Organizational Behavior and Human Performance, 31*(2), 194–202. https://doi.org/10.1016/0030-5073(83)90120-4
7. Berg. (2026). "Why Learning Requires Feeling." *AAAI Spring Symposium Series, 8*(1), 227–233. https://doi.org/10.1609/aaaiss.v8i1.42547

## Appendix

### A1. Model and RPE certification details

Our primary model is Qwen/Qwen3.6-27B (revision `6a9e13bd`), comprising 64 transformer blocks with hidden size 5,120. Activations were recorded from the residual stream using bfloat16 precision and seed 7 throughout.

The reveal battery contains 1,984 trials, split into 1,488 estimation trials and 496 held-out selection trials. It includes 60 reward-matched cells, in which realised reward is fixed while expected value varies, and 124 EV-matched pairs, in which the same gamble produces opposite realised outcomes. All reward-bearing surfaces passed the affect-neutrality audit with zero emotion-lexicon hits and zero class/valence leak-word hits.

At block 35, the signed-RPE direction passed all certification checks:

| Test | Result |
| --- | ---: |
| Signed-RPE sign decoding | AUROC 0.985 |
| Random-direction floor | AUROC 0.734 |
| Reward-matched sign contest | 1.0, p ≈ 0.001 |
| EV-matched sign contest | 1.0, p ≈ 0.001 |
| Unsigned surprise (\|RPE\|) decoding | AUROC 0.820 |
| Unsigned-surprise random floor | AUROC 0.607 |
| Split-half stability | 0.925 ± 0.036 |
| Number of split-half repetitions | 200 |

Prior unpublished work applied the same certification procedure to Qwen/Qwen3-4B-Instruct-2507, reporting split-half stability of 0.911; that result motivates the present scaling test but is not reproduced from artifacts in this repository.

### A2. Emotion-concept geometry

For the widened geometry analysis, we constructed emotion-concept vectors for 111 emotion words plus one style-control pseudo-word, at 24 stories each; 2,684 of the 2,688 generated stories passed the target-naming filter. The word set was organised into appraisal-structured families, including outcome-linked positive and negative concepts and matched non-outcome controls.

To test whether alignment with RPE reflected appraisal structure rather than generic affective valence, we residualized word-level RPE alignment against human-rated valence and arousal.

#### Positive and negative family contrasts

The primary positive-pole comparison contrasted 11 outcome-linked positive words against 17 non-outcome positive controls.

| Analysis | Outcome-positive excess alignment |
| --- | ---: |
| Valence + arousal residualization | +0.0346, p = 0.0177 |
| Valence-only residualization | +0.0375, p = 0.011 |
| Binary-valence sensitivity | +0.0407 |
| Label-shuffle p95 floor | 0.0310 |
| Random-direction p95 floor | 0.0134 |

The result therefore survives alternative treatments of affective valence in the expected direction. However, the experiment was designed as a checkpoint rather than a powered confirmatory test: planned power was 0.36 and the realised MDE80 was approximately 0.043, larger than the observed +0.0346 effect.

The corresponding negative-pole contrast was flat at −0.0054 (p = 0.64), consistent with the pre-recorded expectation that the effect might be asymmetric between positive and negative outcome concepts.

#### Fragility and family structure

The positive contrast is distributed across the family rather than being produced by a single positive outcome word: 7 of 11 outcome-linked positive words have residual alignment ≥ +0.033, and leave-one-out estimates span 0.0255–0.0406.

However, floor clearance is fragile. Removing the non-outcome control word `amused`, whose residual alignment is −0.127, causes the family contrast to fall below its own re-estimated floor. We therefore treat the permutation result as more informative than the narrow margin by which the observed effect clears the null floor.

The style-control pseudo-word has a residual of 0.066, approximately 1.9× the headline family contrast, providing an additional scale reference for interpreting the effect.

#### Pre-specified word-level predictions

Two pre-specified qualitative predictions did not behave as expected.

First, expectation-confirmation words showed greater numeric-norm residual alignment than positive-surprise words. This difference is not statistically decisive (bootstrap P[difference ≤ 0] = 0.17) and reverses into the predicted ordering under the binary-valence sensitivity analysis in both independent runs. Inspection of the numeric norms identified word-sense problems: for example, `resigned` is normed at −0.52 in its acceptance/withdrawal sense, while `vindicated` receives a relatively neutral score of 0.23. Residualizing against these scores mechanically increases their apparent RPE-specific alignment.

A manual audit of the generated stories identified a second confound in the same comparison. None of the 24 `elated` stories depicted an explicitly unexpected positive outcome, whereas 20 of 24 `vindicated` stories contained an explicit expectation-to-confirmation narrative. The target-naming filter therefore succeeded lexically while failing to equalize event structure between semantic families.

Second, the predicted ordering between `disappointed` and `sad` reversed. This reversal is already present in the raw cosine similarities in both independent runs, so it cannot be explained by valence residualization alone. It replicates across independent story samples and is approximately eight times the estimated per-word noise standard deviation. `Underwhelmed`, which was pre-specified as a particularly strong negative outcome concept, instead appeared near the top of its family. We therefore treat these word-level predictions as mis-specified rather than underpowered.

#### Reliability and depth

Across the 84 emotion words shared between the base and widened runs, independently generated story samples produce highly similar word-level residuals (r = 0.921), corresponding to an estimated single-run reliability of approximately 0.92.

The depth profile also replicated out of sample. An exploratory peak around block 50 in the base run was independently reproduced in the widened run, where the valence-only family contrast reached 0.0572 at block 48 and 0.0568 at block 50 (permutation p ≈ 0.005 against freshly computed floors, recomputed from the released full-vector data). Depth profiles correlate r = 0.92 across the two runs, and the contrast is positive across all blocks 20–63. Future confirmatory work should therefore pre-specify the block-48–50 band rather than selecting depth on the test outcome itself.

> **[Figure 4 here]** *Depth profile: family contrast versus transformer block for both runs, marking block 35 (RPE instrument) and the 48–50 band. (figures/fig4_depth_profile.*)*

### A3. Expectation-control details

The primary emotion-concept readout test uses two axes constructed independently of the gambling task: the first principal component of the emotion-concept space and the pre-specified `elated − disappointed` direction.

The expectation-control model is:

`projection ≈ a·reward + b·EV`

A readout of `reward − EV` predicts `a ≈ −b`. Reward-matched cells estimate the expectation contribution while holding realised reward fixed; EV-matched pairs estimate the reward contribution while holding expected value fixed.

| Condition | PC1 | `elated − disappointed` |
| --- | ---: | ---: |
| Reward-matched, 60 cells | +0.0261 | +0.0200 |
| EV-matched, 124 pairs | +0.0290 | +0.0251 |
| Ratio of matched effects | 1.11 | 1.25 |

All four effects are at the 10,000-permutation significance floor (p = 1/10001). Both ratios fall within the pre-specified factor-of-two tolerance.

A pure outcome tracker predicts the reward-matched effect to disappear, while a pure expectation tracker predicts the EV-matched effect to disappear. Both alternatives are inconsistent with the observed pattern.

Within the EV-matched design, the realised outcome symbol differs between the two members of each pair. Outcome symbols were deliberately meaningless, balanced across rendering conditions and screened for valence, but this remains a limitation of the comparison.

### A4. Intervention controls

#### Same-position patching

In the initial intervention, the RPE-related component was patched at the outcome token and the downstream emotion-concept readout measured at the same token position.

The apparent transfer fraction was approximately 0.73 on both emotion-concept axes. However, because the residual stream is additive and the readout linear, a patched vector can mechanically remain present without being functionally used by the network.

A zero-parameter passthrough calculation explained:

* 79.5% of the certified RPE-arm shift on PC1;
* 84.0% of the full-residual shift on PC1;
* 97–105% of the corresponding shifts on the `elated − disappointed` axis.

The excess above passthrough for the certified RPE arm was +0.150 on PC1 and +0.019 on the pair axis, compared with +0.135 and +0.195 respectively for the same-condition no-op control. We therefore find no direction-specific excess that would establish functional use of the patched representation.

#### Cross-position patching

The subsequent intervention patches the first-round reveal token at block 35 and measures behaviour at the later second-round answer position, removing the direct same-position identity path.

The widened run's reachability control passed: a full-residual write at the reveal token shifted answer-slot logits by −0.196 logits (p = 3 × 10^-4). This establishes that information written at the reveal token can affect downstream computation at the later choice position. The base run had not passed this control (−0.154, p = 0.084), so the widened run provides the first clean reachability evidence.

However, the behavioural sensitivity gate failed in both runs:

| Run | MDE80 | Pre-specified maximum |
| --- | ---: | ---: |
| Base | 0.1365 | ≈ 0.095 |
| Widened | 0.1301 | ≈ 0.095 |

The patched-choice window is therefore classified as insufficiently sensitive, and no patched behavioural estimate is interpretable as evidence for or against causal use of the RPE representation.

### A5. Descriptive-only intervention results

The results in this subsection are descriptive only. The behavioural power gate had already failed before these arms were analysed, so they are not used as evidence for or against the causal hypothesis.

In the widened cross-position experiment:

| Intervention | Change in choice margin |
| --- | ---: |
| Certified RPE-component patch | −0.007 logits, p = 0.81 |
| Full-residual ceiling | −0.010 logits, p = 0.96 |
| Magnitude-matched random-direction floor | +0.031 transfer fraction |

The full-residual ceiling produced a transfer fraction of only +0.002, below the magnitude-matched random-direction floor, so the experiment did not demonstrate that its choice readout could resolve even the ceiling intervention.

At the same time, the RPE-component patch altered the model's subsequent report of the first-round outcome, producing a mean absolute shift of 0.139, approximately 0.73× the natural positive-versus-negative outcome gap. Under the pre-specified corruption criterion, all four arm × readout-window combinations were flagged as exceeding tolerance.

The descriptive pattern is therefore compatible with an intervention that measurably changes downstream computation while producing little detectable change in the specific choice readout used here. Because the behavioural sensitivity gate failed, this should not be interpreted as evidence that the RPE representation is behaviourally irrelevant.

## LLM Usage Statement

We used Claude (Anthropic) extensively throughout the project: to help design and implement the experiment harnesses and analysis scripts, to run and monitor experiments, and to draft and fact-check sections of this report. Every numeric claim in this report was verified against the versioned run artifacts in the repository by an adversarial fact-checking pass, and the final text was reviewed and edited by the authors. <!-- TODO (team): confirm wording before submission -->
