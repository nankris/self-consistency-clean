# 01: The pilot gate was relaxed after it failed

- **Experiment:** this repo
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**The pilot gate was relaxed after it failed.** results/runner.log lines 66-73: "Pilot complete: greedy parse fail rate = 2.0% ... RuntimeError: Pilot failed gate". The session was then restarted (line 93). scripts/colab_runner.py now has `passed: greedy_fail_rate <= 0.02` and the log shows "Pilot already completed and passed" (line 145). The README secondary prediction says "< 2.0%", and Decision 4 admits the <=2% criterion was set after the pilot. The failure was a single length truncation, so this is low stakes, but the blog's "passed cleanly" plus "pre-registered rules do not get rewritten after seeing the data" is not honest given this.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the blog post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
