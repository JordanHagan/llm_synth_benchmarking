# Executive Summary

This technical report presents the evaluation results of two AI/ML models, Model A and Model B, based on an A/B test conducted against a golden dataset. The assessment focuses on the quality of translation/response and accuracy of generated responses, utilizing BLEU scores, Word Error Rate (WER), and other supplementary metrics.

Model A outperformed Model B in terms of BLEU score and task completion, demonstrating better overall translation/response quality and accuracy. However, Model B showed slightly better performance in response relevance and clarity.

# Methodology

## Feature Configuration

The A/B test involved two models, Model A and Model B, with 15 test cases per model, each model running through three rounds to account for the probabilistic output of large language models. JSON tests were not enabled, and validation was enabled throughout the test.

## Test Approach

The test approach consisted of running both models against the same dataset, comparing the generated responses using predefined evaluation metrics. Each model was tested three times for each test case to ensure a robust and reliable comparison.

# Metrics Analysis

## BLEU Score & WER Analysis

Model A achieved an average BLEU score of 0.05958, compared to Model B's score of 0.05428. Model A's average WER score was 2.448, versus Model B's score of 1.275. In relative terms, Model A exhibited a 10.2% higher BLEU score and a 91.0% higher WER score compared to Model B. These findings indicate that Model A produces responses of higher quality but with lower accuracy than Model B.

## Additional Metrics

Model A demonstrated higher response relevance (9.68% vs. 11.28%) and lower clarity (93.19% vs. 95.31%) compared to Model B. Model A also outperformed Model B in average task completion (91.5% vs. 71.7%). These results suggest that Model A generates more relevant but less clear responses, while Model B provides less relevant but clearer outputs.

# Results & Findings

Model A outperformed Model B in terms of BLEU score and task completion, indicating better overall translation/response quality and accuracy. However, Model B showed better performance in response relevance and clarity, suggesting that Model B might be more suitable for specific applications requiring clearer outputs.

# Recommendations

Considering the overall performance, it is recommended to use Model A as the primary model due to its superior BLEU score and task completion performance. However, for applications requiring clearer outputs, Model B may be a more suitable alternative. It is crucial to weigh the trade-offs between response quality and clarity when selecting the most appropriate model for specific use cases.

Further investigation and tuning of both models may lead to improved performance, particularly in the areas where Model B currently outperforms Model A. Exploring techniques to balance response quality and clarity is an essential area for future research.