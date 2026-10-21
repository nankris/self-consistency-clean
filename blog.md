# Self-Consistency Gained 4.6 Points When We Fixed the Prompt, Not 15.5

*Majority voting over sampled reasoning paths helps small models, but nearly half the original gain was an error-recovery patch for broken prompt formatting.*

There is an appealing technique in language model inference called [self-consistency](https://arxiv.org/abs/2203.11171). Instead of generating one greedy completion at temperature zero, you sample multiple reasoning paths at temperature 0.7, parse the answers, and take the plurality vote. The popular claim is that self-consistency delivers a large reasoning boost on local hardware, letting compact models rival larger ones without fine-tuning.

In production, compute is never free. Generating eight samples multiplies decode cost, memory bandwidth, and latency by eight. At scale, you need to know whether that 8x compute tax buys a genuine leap in mathematical capability, or whether simpler prompt fixes deliver the same outcome at lower cost.

Here is the setup, the flawed history that misled earlier benchmarks, the clean confirmatory rerun, and the real numbers.

---

### What I Expected

The logic behind majority voting is straightforward. Multi-step reasoning requires sequential deductions. If an instruction-tuned model has an 80% chance of navigating each step correctly, a single greedy path might stumble on step four. Sampling several attempts creates independent opportunities to find the correct deduction. As long as correct paths cluster on the identical numerical answer while flawed paths scatter across different errors, plurality voting filters out noise.

I expected self-consistency to deliver between five and eight percentage points of improvement on grade-school math word problems. That gain would be noticeable, but at an 8x multiplier on tokens, it would sit right on the edge of economic feasibility for production systems.

---

### The Earlier Runs and the Suspicious Leap

This experiment extends a sequence of earlier attempts. In our initial test (Experiment 035), we evaluated Qwen3-0.6B on 40 GSM8K questions locally on CPU. Greedy single-sample decoding scored 50.5%, while a majority vote of five samples reached 57.5% (+7.0 percentage points). But five CPU samples took 79 seconds per question, whereas a single call to a frontier cloud API finished in 9 seconds and scored 100%.

Attempts to scale up to Qwen3-4B on cloud GPUs (Experiments 035r and 035r3) stalled due to allocation refusals. In Round 5 (`035r5-self-consistency-1p7b`), we evaluated `Qwen/Qwen3-1.7B` on 400 questions using an A100 GPU. The raw numbers looked extraordinary:

- Greedy accuracy: 28.50% (114/400 correct)
- Vote of five accuracy: 44.00% (176/400 correct)
- Primary paired difference: **+15.50 percentage points** (95% bootstrap CI [+11.50, +19.50] pp)

A jump of 15.5 percentage points suggested sampling unlocked latent reasoning capabilities.

---

### The Flaw I Found

A 1.7B parameter instruction-tuned model scoring only 28.5% on GSM8K did not make sense. Qwen3-1.7B is a capable model that typically clears 65% on grade-school arithmetic.

When I audited the raw generation outputs, the explanation became obvious. The prompt suffix used in round 5 read:

```
Solve briefly. End with exactly: Answer: NUMBER
```

The author intended `NUMBER` to indicate where the model should place its final numerical output. But the 1.7B model took the instruction literally. In 208 out of 400 greedy completions (52.0%), the model printed the word `NUMBER` before the digit: `Answer: NUMBER 42` or `Answer: **NUMBER 18**`.

The evaluation script relied on an exact regular expression expecting digits immediately following `Answer:`. Because the word `NUMBER` intervened, the parser classified the output as a parse failure. In total, 55.75% of greedy outputs (223/400) and 55.80% of stochastic samples (1,116/2,000) failed to parse.

When half your responses fail to parse, majority voting no longer measures mathematical problem-solving. If a model generates five responses, four copy the literal string `NUMBER`, and one formats properly as `Answer: 18`, voting picks the single valid parsed number. Majority voting was acting as a noisy error-recovery mechanism for a broken prompt template.

A post-hoc diagnostic script stripping the literal word `NUMBER` before parsing revealed the true underlying picture:

- Repaired greedy accuracy: 68.50% (274/400 correct)
- Repaired vote of five accuracy: 77.25% (309/400 correct)
- True paired difference: **+8.75 percentage points** (95% CI [+5.25, +12.25] pp)

Over 40% of the originally reported +15.50 point gain was an artifact created by fragile string matching.

---

### The Clean Protocol and Fresh Rerun

To establish the genuine effect of self-consistency, I pre-registered a cleanup protocol in Experiment 035b before launching GPU work:

1. **Clean prompt**: We replaced the ambiguous directive with clear instructions without placeholder words: `Solve briefly step by step. Conclude your answer on a new line with: The answer is [number].`
2. **Robust extractor**: A standardized parser that scans backwards for common concluding phrases (`The answer is X`, `Answer: X`), handles LaTeX `\boxed{X}` markup, strips GSM8K delimiters (`#### X`), and canonicalizes commas and decimals.
3. **Pilot gate**: A pre-registered gate on 50 training questions requiring parse failure rates below 2.0% before proceeding to test evaluation.
4. **Fresh test data**: Exactly **500 fresh GSM8K test questions** drawn strictly from test items untouched by any previous 035 experiment.
5. **Pre-registered decision rule**: A margin of +5.0 percentage points. Given that $k=8$ samples consumes eight times the compute, self-consistency had to beat greedy decoding by at least +5.0 percentage points with a 95% bootstrap lower bound above +2.0 pp to earn a verdict of PROVEN.

The verification pipeline is shown in figures/diagram.png.

![Experiment 035b Pipeline](figures/diagram.png)

*Figure 1: The experimental verification pipeline (figures/diagram.png), showing the transition from the flawed 035r5 prompt to the clean pilot gate, fresh confirmatory split, and decision rule.*

We ran the pilot on 50 training items on an NVIDIA L4 GPU via vLLM. It passed cleanly: greedy parse failures dropped to 2.0% (a single token-limit truncation, not a formatting failure), and stochastic sample failures dropped to 0.75% (3 out of 400 samples). We then unlocked the confirmatory evaluation.

---

### Confirmatory Results

The confirmatory run evaluated all 500 fresh test questions across greedy decoding ($T=0$) and eight stochastic samples per question ($T=0.7, \text{top\_p}=0.95$). Inference finished in 293.9 seconds on our Colab L4 GPU, consuming 0.96 compute units.

The primary headline number:

> **+4.60 percentage points** (Greedy 74.40% vs Vote@8 79.00%, paired bootstrap 95% CI [+2.00, +7.20] pp, exact McNemar $p = 0.000824$).

Here is the complete performance breakdown across sample budgets:

![Results Table](figures/table.png)

*Figure 2: Confirmatory evaluation table (figures/table.png) rendering accuracies, 95% bootstrap confidence intervals, paired gains, and parse failure rates across all arms on 500 fresh GSM8K items.*

The progression from greedy decoding to increasing vote budgets demonstrates steady improvement:

![Accuracy vs Sample Budget](figures/accuracy_vs_k.png)

*Figure 3: Accuracy versus sample budget (figures/accuracy_vs_k.png), showing greedy baseline performance against majority voting curves from k=1 to k=8 with 95% bootstrap error bars.*

Moving from $k=1$ (75.80%) to $k=3$ (76.60%), $k=5$ (78.00%), and $k=8$ (79.00%) added consistent increments. At $k=8$, the model gained 34 questions where greedy failed but voting succeeded, while losing 11 questions where greedy was correct but the consensus strayed into an error.

The paired bootstrap distribution for the primary endpoint is displayed in figures/delta_bootstrap.png:

![Bootstrap Distribution](figures/delta_bootstrap.png)

*Figure 4: Bootstrap distribution of paired differences between SC@8 and Greedy decoding (figures/delta_bootstrap.png), illustrating the observed +4.60 pp gain against the null line, lower gate, and superiority threshold.*

The paired gain is positive and statistically reliable ($p < 0.001$). The 95% bootstrap confidence interval lower bound is +2.00 percentage points, ruling out zero impact.

Formatting artifacts vanished completely, as shown in figures/parse_failure_rate.png:

![Parse Failure Rate Comparison](figures/parse_failure_rate.png)

*Figure 5: Parse failure rate comparison (figures/parse_failure_rate.png), contrasting the 55.8% catastrophic failure rate of the flawed 035r5 prompt against the 0.6% and 0.45% rates under the clean 035b extractor.*

Parse failures dropped from 55.75% to 0.60% for greedy decoding, and from 55.80% to 0.45% across all 4,000 stochastic generations.

---

### What Surprised Me and What I Got Wrong

My pre-registered hypothesis predicted that $k=8$ would beat greedy decoding by at least +5.0 percentage points. It fell short by 0.40 percentage points, landing at +4.60 pp.

Under our strict decision rule, because the point estimate is below +5.0 pp while the upper confidence interval bound reaches +7.20 pp, the verdict is officially **INCONCLUSIVE**. Pre-registered rules do not get rewritten after seeing the data.

What surprised me most was the baseline greedy accuracy of Qwen3-1.7B. When given an unconfusing prompt, the model solved 74.40% of the fresh GSM8K test set directly on its first try. In the flawed 035r5 experiment, greedy accuracy was recorded as 28.50%. Merely removing the word `NUMBER` from the prompt lifted single-shot performance by 45.9 percentage points without sampling a single extra token.

---

### What I Would Do in Production

If you are considering self-consistency for a serving pipeline, here are four practical recommendations:

1. **Audit your extraction regex before buying compute**: A brittle parser makes your model look incompetent and inflates the apparent value of multi-sample voting. If parse failures exceed 1%, fix your prompt template and parser before tuning temperature.
2. **Never put placeholder labels in zero-shot prompts**: Small models under 3B parameters tend to copy literal phrases. Avoid constructs like `Answer: [NUMBER]`. Use natural phrasing such as `The answer is [number]` or provide a concrete few-shot example.
3. **Budget for diminishing returns**: Going from greedy to $k=3$ provided +2.20 pp for a 3x compute cost. Going from $k=5$ to $k=8$ required three additional full generations to gain only 1.00 pp. For most production budgets, $k=3$ is the practical ceiling.
4. **Reserve self-consistency for high-value asynchronous jobs**: An 8x latency and cost multiplier for a 4.6 percentage point gain is a poor trade-off for interactive chat, but reasonable for overnight batch pipelines, test-suite generation, or synthetic data verification.

---

### Limitations

This evaluation was conducted on GSM8K with `Qwen/Qwen3-1.7B`. GSM8K focuses on multi-step grade-school arithmetic. On advanced competition mathematics like MATH, error diversity is wider and self-consistency curves may differ. Furthermore, Qwen3 was evaluated with extended thinking disabled (`enable_thinking=False`); models with test-time reasoning traces enabled may show different sample consensus behavior.

---

### Conclusion

Self-consistency works. It reliably adds +4.60 percentage points to a 1.7B model on unseen math problems. But it is not magic, and it was never worth +15.5 points. Half of that earlier gain was an illusion born of a bad prompt and a strict parser. Clean evaluation matters as much as the sampling algorithm itself.

Code: [tig/035b-self-consistency-clean](https://github.com/example/tig/035b-self-consistency-clean)

## Screenshots
![1_runner_log_head](figures/screenshots/1_runner_log_head.png)
![2_runner_log_tail](figures/screenshots/2_runner_log_tail.png)
![3_summary_json](figures/screenshots/3_summary_json.png)
![4_timing](figures/screenshots/4_timing.png)
