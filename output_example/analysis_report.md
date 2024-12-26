# Executive Summary

This technical report presents the evaluation results of two AI/ML models, Model A and Model B, based on an A/B test against a golden dataset. The test was conducted with a total of 15 test cases, with each model generating responses three times for each prompt. The primary focus of this evaluation is on BLEU scores, Word Error Rate (WER), and other supplementary metrics.

Model B outperformed Model A in BLEU score, with an average BLEU score of 0.04608 compared to Model A's 0.04233. However, Model A had a higher average task completion rate of 0.625 compared to Model B's 0.545. Model B also showed a lower WER of 1.310 compared to Model A's 2.481.

# Methodology

## Feature Configuration

The following features were tested:

- Model A
- Model B

A total of 15 test cases were conducted, with each model generating responses three times for each prompt.

## Test Approach

The test approach involved an A/B test of the two models against a golden dataset. Responses were generated 3 times for each prompt and then combined together for analysis to account for the probabilistic output of LLMs within text generation. The primary metrics used for evaluation are BLEU scores, WER, response relevance, clarity, and task completion.

# Metrics Analysis

## BLEU Score & WER Analysis

Model B outperformed Model A in BLEU score, with an average BLEU score of 0.04608 compared to Model A's 0.04233. This indicates that Model B produced translations/responses that were closer to the reference translations/responses in the golden dataset.

Model B also showed a lower WER of 1.310 compared to Model A's 2.481. This indicates that Model B had fewer word errors in its translations/responses compared to Model A.

## Additional Metrics

Model A had a higher average task completion rate of 0.625 compared to Model B's 0.545. This indicates that Model A was able to complete more tasks successfully compared to Model B.

Model B, however, had a higher response relevance score of 0.108 compared to Model A's 0.073. This indicates that Model B's responses were more relevant to the prompts given.

Model B also had a higher clarity score of 0.950 compared to Model A's 0.919. This indicates that Model B's responses were clearer and easier to understand compared to Model A's.

# Results & Findings

Model B outperformed Model A in BLEU score, WER, response relevance, and clarity. However, Model A had a higher task completion rate compared to Model B.

The higher BLEU score and lower WER of Model B indicate that it produced more accurate translations/responses compared to Model A. The higher response relevance and clarity scores of Model B indicate that its responses were more relevant and clearer compared to Model A.

However, Model A's higher task completion rate indicates that it was able to complete more tasks successfully compared to Model B. This may be due to Model A being more conservative in its responses, leading to a higher success rate in completing tasks.

# Recommendations

Based on the results of this evaluation, Model B is recommended for use in cases where accuracy in translation/response and response relevance are critical. However, if task completion is a key requirement, Model A may be a better choice due to its higher task completion rate.

It is recommended that both models continue to be evaluated and improved over time, as advancements in AI/ML technology may result in improved performance. Additionally, it is recommended that other metrics, such as human evaluation, be used in conjunction with automated metrics to provide a more comprehensive evaluation of the models.