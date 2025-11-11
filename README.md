# Customer Review Prompt Engineering Analysis

A comprehensive study exploring prompt engineering techniques for analyzing customer reviews using foundation models (GPT-4.1-nano and Gemini Flash). This project demonstrates iterative prompt refinement across three core NLP tasks: sentiment classification, key phrase extraction, and review summarization.

## Overview

This project systematically evaluates how different prompt engineering strategies affect model performance on customer review analysis tasks. Using the IMDb Pulp Fiction reviews dataset, I developed and tested three progressive versions of prompts for each task, comparing results from OpenAI's GPT-4.1-nano and Google's Gemini Flash.

## Key Features

- **Three Core Tasks**: Sentiment classification, key praise/complaint extraction, and two-sentence summarization
- **Iterative Prompt Engineering**: Three versions (v1-v3) per task, each building on previous insights
- **Multi-Model Comparison**: Side-by-side evaluation of GPT-4.1-nano and Gemini Flash
- **Comprehensive Evaluation**: Uses Macro-F1, Token-level F1, and ROUGE-L metrics
- **Cost-Aware Implementation**: Includes API throttling and quota management for budget optimization

## Repository Structure

```
├── data/
│   └── Pulp Fiction (Movie) - Imdb Reviews (250).csv
├── results/
│   ├── sentiment_predictions_all_250.csv
│   ├── reviews_with_snippets.csv
│   ├── reviews_with_summaries.csv
│   ├── snippet_leaderboard.csv
│   ├── summary_leaderboard.csv
│   └── [various subset analysis files]
├── customer_review_prompt_engineering.ipynb
├── .gitignore
└── README.md
```

## Methodology

### Dataset
- **Source**: IMDb Pulp Fiction reviews (250 reviews)
- **Validation**: 20-review manually labeled subset for iterative refinement
- **Full Evaluation**: Complete dataset for final pipeline testing

### Prompt Development Strategy

For each task, I developed three successive prompt versions:

1. **Version 1 (Baseline)**: Simple, plain English instructions
2. **Version 2 (Rule-Based)**: Added explicit rules, keywords, and constraints
3. **Version 3 (Few-Shot)**: Incorporated examples when rule-based tuning plateaued

### Evaluation Metrics

- **Sentiment Classification**: Macro-F1 (balances performance across Positive, Neutral, Negative)
- **Phrase Extraction**: Token-level F1 (rewards partial matches)
- **Summarization**: ROUGE-L F1 (measures n-gram overlap with gold summaries)

## Results

### Best-Performing Combinations

| Task | Best Model + Prompt | Metric | Score | Latency | Cost |
|------|-------------------|--------|-------|---------|------|
| **Sentiment** | GPT-4.1-nano + v2 | Macro-F1 | **0.589** | ~0.8s/call | ~$0.002/1K tokens |
| **Phrase Extraction** | GPT-4.1-nano + v3 | Token-F1 | **0.503** | ~0.8s/call | ~$0.002/1K tokens |
| **Summarization** | Gemini Flash + v2 | ROUGE-L | **0.216** | ~0.2s/call | Free tier |

### Task-Specific Findings

#### Sentiment Classification
**GPT-4.1-nano with v2** achieved the best balance (Macro-F1: 0.589) by:
- Explicitly flagging negative keywords ("worst", "skip it", "don't waste")
- Including rules for sarcasm detection
- Handling mixed praise/criticism with neutral label guidelines

**Key Challenge**: Models struggled with neutral reviews containing balanced opinions.

#### Key Phrase Extraction
**GPT-4.1-nano with v3** (Token-F1: 0.503) succeeded through:
- Few-shot examples showing ideal extraction patterns
- "Strongest emotional language" selection rule
- Complete sentence requirements with punctuation validation

**Key Challenge**: Occasional truncation when emotional sentences exceeded generation limits.

#### Summarization
**Gemini Flash with v2** (ROUGE-L: 0.216) excelled due to:
- Strict two-sentence, ≤35-word constraint
- Paraphrasing requirement (avoiding copy-paste)
- Focus on key reasons (acting, pacing, dialogue)

**Key Challenge**: Tendency toward formulaic phrasing ("A glowing review praising...").

## Failure Patterns & Root Causes

### Common Issues Across Tasks

1. **Edge-Class Struggles**: Models consistently mishandled ambiguous or minority-class inputs
   - Neutral sentiment reviews were frequently mislabeled as positive/negative
   - Balanced reviews challenge both prompt rules and model priors

2. **Sarcasm & Irony**: Highly ironic phrasing sometimes flipped sentiment labels despite explicit rules

3. **Length Violations**: Strict word/token caps occasionally forced:
   - Truncated snippets mid-sentence
   - Loss of critical details in summaries

4. **Template Bias**: Few-shot examples sometimes induced formulaic language over genuine paraphrasing

## Metric Limitations

- **Small Sample Size**: 20 manually labeled examples means single errors shift scores by ~5%
- **Token-F1 Bias**: Over-rewards short snippets with coincidental word overlap
- **ROUGE-L Gap**: Measures n-gram overlap, not semantic fidelity or factual accuracy
- **Mitigation**: Supplemented metrics with manual failure case analysis

## Cost Optimization

The project implements a **quota-aware workflow**:
1. **Gemini-First Strategy**: Use free-tier Gemini Flash for initial passes
2. **Rate Limiting**: Custom throttling (9-14 RPM) respects API quotas
3. **Selective OpenAI**: Only invoke GPT-4.1-nano for completed/critical prompts
4. **Result**: ~60% reduction in total API costs

## Setup & Installation

### Prerequisites
```bash
pip install pandas openai google-generativeai scikit-learn rouge-score tqdm jupyter
```

### API Configuration
1. Obtain API keys:
   - [OpenAI API Key](https://platform.openai.com/api-keys)
   - [Google Gemini API Key](https://makersuite.google.com/app/apikey)

2. Create credential files (excluded from git):
   ```
   openai_api.txt    # Contains your OpenAI API key
   gemini_api.txt    # Contains your Gemini API key
   ```

### Running the Analysis
```bash
jupyter notebook customer_review_prompt_engineering.ipynb
```

## Key Insights & Lessons Learned

### 1. Minimalism Wins Early
Short, rule-based prompts (v2 variants) often outperformed longer few-shot prompts on skewed class distributions by focusing models on critical decision factors.

### 2. Few-Shot is Powerful but Perilous
- **Pro**: Clarifies expected output format and style
- **Con**: Over-reliance can induce template bias or override essential rules
- **Best Practice**: Use sparingly, with diverse examples

### 3. Metric Choice Matters
- Exact-match accuracy painted a pessimistic picture (<0.2) for snippet extraction
- Token-level F1 revealed nuanced partial matches (>0.5)
- **Recommendation**: Always supplement automated metrics with human assessment

### 4. One Prompt Doesn't Fit All
- **Sentiment & Extraction**: Thrived on straightforward rule lists
- **Summarization**: Required strict sentence/word-count caps + paraphrasing constraints
- **Implication**: Task-specific optimization essential

### 5. Model Trade-offs
- **GPT-4.1-nano**: More consistent across tasks, better structured extraction
- **Gemini Flash**: 4× faster, cost-free (free tier), but weaker on extraction tasks

## Deployment Confidence: Grade B

**Suitable For**:
- Dashboard aggregation of customer feedback
- User-facing review snippets
- First-pass sentiment triaging

**Requires**:
- Human review of neutral sentiment cases
- Periodic spot-checking of summary accuracy
- Post-generation filters for truncation detection

## Future Work

1. **Fine-Tuning**: Train small open-weights model for improved neutral recall
2. **Human Fluency Scores**: Add 1-5 ratings for summary quality beyond ROUGE
3. **Post-Generation Heuristics**: Implement lightweight filters for:
   - Truncation detection (sentence completeness)
   - Template language detection
   - Minimum snippet length validation
4. **Expanded Dataset**: Test on 500+ reviews with stratified class sampling

## Dataset Citation

IMDb Pulp Fiction reviews dataset from [Kimola NLP Datasets](https://github.com/Kimola/nlp-datasets).

## License

This project is open source and available for educational and research purposes.

## Author

**Usama Ali**
[GitHub](https://github.com/aaliusama) | [LinkedIn](https://linkedin.com/in/aaliusama)

---

*This project demonstrates practical prompt engineering techniques for real-world NLP tasks, with emphasis on cost-efficiency, iterative refinement, and transparent evaluation.*
