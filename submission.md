# Submission

## Page 1 — Abstract + question

Reward prediction error (RPE) theory holds that organisms evaluate rewards relative to prior expectations (Schultz, Dayan & Montague, 1997). In humans, reward prediction errors have also been shown to predict momentary subjective well-being, such that outcomes that exceed expectations are associated with greater happiness than equivalent outcomes that were already expected (Rutledge et al., 2014). We apply RPE theory to Qwen/Qwen3.6-27B and {INSERT ABBREVIATED DESCRIPTION OF METHOD HERE}. We find that {INSERT FINDINGS}.

## Pages 1–2 — Introduction

In the sense that a large language model (LLM) is a digital mind it may carry analogies to biological minds in how it computes well-being and how that computation influences its subsequent behaviour. Along this line of reasoning in this paper we set out to test the following hypotheses:

- emotion-concept readout at the outcome token tracks the model's reward-prediction error
- recently computed RPE influences a model's follow-up

As LLMs increasingly take actions in the world through agentic systems, understanding the underlying mechanisms of those actions can yield insights into pre-existing biases and give ways of counteracting them.

## Pages 2–3 — Method

First, we set out to prove that RPE computations scale from a previously researched smaller Qwen/Qwen3-4B-Instruct-2507 model to the larger Qwen/Qwen3.6-27B, which is the base of subsequent experiments. The test battery consists of 1984 deliberately neutral scenarios describing a 50/50 gamble between two outcomes, followed by a reveal of which outcome actually occurred. The model's internal activations are recorded at the exact moment the outcome is revealed, to test whether a clean, separable RPE signal exists in its internal representations. Outcome labels are meaningless three-letter codes and point values are round numbers, with the framing sentence rotated across several neutral phrasings, so the design contains no emotional language that could confound the results.

We then build an emotion map.

We use Qwen/Qwen3.6-27B as our research model

- Scale the initial RPE research to a larger model to prove validity
- Build and QA emotion map
- why 84 words instead of the initial 171
- remove "non-useful elements"
- consider the "surprise"
- compare against human annotated pleasantness data for QA
- Observe RPE impact
- change the RPE

Our work builds on currently unpublished peer-reviewed research by Hugo Nguyen indicating Qwen/Qwen3-4B-Instruct-2507 performs RPE calculations according to "A Computational and Neural Model of Momentary Subjective Well-Being" (Rutledge, Skandali, Dayan & Dolan, 2014). Furthermore, we rely on the methodology from "Emotion Concepts and their Function in a Large Language Model" (Sofroniew et al., 2026) to construct an emotion map. Finally, we rely on "Gambling with the House Money and Trying to Break Even: The Effects of Prior Outcomes on Risky Choice" (Thaler & Johnson, 1990) and "The Effect of Positive Feelings on Risk Taking: When the Chips Are Down" (Isen & Patrick, 1983) to construct the experiment for impact of prior RPE computations on subsequent actions.

Our code is publicly available here: https://github.com/SystemicVoid/appraisal-emotions

## Pages 3–5 — Main results

- Only genuinely new outcomes, prior research to be explained in the introduction
- Answers to the 3 hypotheses + discussion

## Pages 5–6 — Controls / robustness

- This is where we talk about the probe that tests whether RPE really represents a worse/better vector // change the gamble (stage 4)
- Should there maybe be another one 🤔

## Page 6–7 — Interpretation + alternative explanations

Aligned against the hypotheses with the following structure:

- **What we observed:** Models exhibit behavioural pattern X.
- **What this supports:** Pattern X appears sufficiently stable/context-resistant/etc. to satisfy our operational definition Y.
- **What it does not establish:** That the model literally experiences desire, suffering, welfare, consciousness, etc.

## Page 7 — Limitations

- Single smallish model
- what else

## Page 8 — Conclusion

- Three paragraphs maximum.
- Follow-on research
- self-reporting as a way to QA results
- possibly "surprise" if we decide to take it out
- possibly patching in the emotion vector rather than the RPE
- Our findings offer representational-level* empirical support for the computational premise of Berg (2026), that emotion-concept representations are organized around goal-relative prediction error.

## Page 9 — Limitations and Dual-Use / Ethical Considerations appendix

Include any risks of over-attributing or under-attributing moral status, and how you handled potentially distressing model outputs. For introspection and preference work, note whether your design establishes a ground-truth or causal link rather than relying on conversation alone.

## Page 10 — References
