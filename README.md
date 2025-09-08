# Pretrain and Pray

-   **Group name:** Pretrain and Pray
    
-   **Group repository:** https://github.com/BerfinTaskin/DL_NLP
    
-   **Tutor responsible:** 	Niklas Bauer
    
-   **Group team leader:** Berfin Taskin
    
-   **Group members:** 
- Berfin Taskin    ->  berfin.taskin@stud.uni-goettingen.de 
- Ashutosh Jaiswal ->  ashutosh.jaiswal@stud.uni-goettingen.de
- Jonathan Henrich ->  jonathan.henrich@uni-goettingen.de
- Emre Semercioglu ->  emre.semercioglu@stud.uni-goettingen.de
- Shrinath Madde   ->  shrinath.madde@stud.uni-goettingen.de

  
# Setup instructions 
Explain how we can run your code in this section. We should be able to reproduce the results you've obtained. 

In addition, if you used libraries that were not included in the conda environment 'dnlp' explain the exact installation instructions or provide a ```.sh``` file for the installation.

Which files do we have to execute to train/evaluate your models? Write down the command which you used to execute the experiments. We should be able to reproduce the experiments/results.

_Hint_: At the end of the project you can set up a new environment and follow your setup instructions making sure they are sufficient and if you can reproduce your results. 

# Methodology

BART GENERATION TASK: 

We implemented a Sample-and-Rerank (Rerank@k) generation strategy that focuses on balancing meaning and diversity in text generation. Here's our approach:

1. **Sample k candidates per input** using stochastic decoding with the following parameters:
   - top-p = 0.9
   - temperature = 0.9
   - no_repeat_ngram_size = 3
   - max_length = 50

2. **Score each candidate** using a non-transformer reward that balances meaning vs. diversity through:
   - Semantic similarity: Using TF-IDF word cosine and TF-IDF character (3-5) cosine, averaged
   - WordNet soft overlap (optional): Giving credit for synonym matches (no context models)
   - Copy penalties:
     - n-gram overlap (1-4) between candidate and source (PINC-style)
     - Self-BLEU(candidate, source) as an additional copying proxy
     - Length penalty: |len(y)-len(x)| / len(x)

3. **Pick the best candidate** by the combined score using the following formula:
   R(x,y) = α·TFIDF_avg(x,y) + ε·WordNet(x,y) - β·Overlap(x,y) - γ·SelfBLEU(x,y) - δ·LenPenalty(x,y)

Default weights used:
- α = 1.0 
- β = 0.30
- γ = 0.20
- δ = 0.10
- ε = 0.20
- k = 10 (also tested with k∈{5,10,20})

This approach maintains rule-compliance by not using extra transformer embeddings while explicitly balancing the trade-off between adequacy and diversity in the generated text.

# Experiments
Keep track of your experiments here. What are the experiments? Which tasks and models are you considering?

Write down all the main experiments and results you did, even if they didn't yield an improved performance. Bad results are also results. The main findings/trends should be discussed properly. Why a specific model was better/worse than the other?

You are **required** to implement one baseline and improvement per task. Of course, you can include more experiments/improvements and discuss them. 

You are free to include other metrics in your evaluation to have a more complete discussion.

Be creative and ambitious.

For each experiment answer briefly the questions:

- What experiments are you executing? Don't forget to tell how you are evaluating things.
- What were your expectations for this experiment?
- What have you changed compared to the base model (or to previous experiments, if you run experiments on top of each other)?
- What were the results?
- Add relevant metrics and plots that describe the outcome of the experiment well. 
- Discuss the results. Why did improvement _A_ perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns?

## Results 
Summarize all the results of your experiments in tables:

| **Stanford Sentiment Treebank (SST)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Quora Question Pairs (QQP)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Semantic Textual Similarity (STS)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Paraphrase Type Detection (PTD)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Paraphrase Type Generation (PTG)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

Discuss your results, observations, correlations, etc.

Results should have three-digit precision.
 

### Hyperparameter Optimization 
Describe briefly how you found your optimal hyperparameter. If you focussed strongly on Hyperparameter Optimization, you can also include it in the Experiment section. 

_Note: Random parameter optimization with no motivation/discussion is not interesting and will be graded accordingly_

## Visualizations 
Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the  different training processes of your improvements in those graphs. 

For example, you could analyze different questions with those plots like: 
- Does improvement A converge faster during training than improvement B? 
- Does Improvement B converge slower but perform better in the end? 
- etc...

## Members Contribution 
Explain what member did what in the project:

**Berfin Taskin:** : Worked with bart generation task both for the first part of the project and the second part of the project(improvement), README file, group leading.

**Ashutosh Jaiswal:** :

**Jonathan Henrich:** :

**Emre Semercioglu:** :

**Shrinath Madde:** :

...

# AI-Usage Card

In the conduct of this research project, we used specific artificial intelligence tools and algorithms ChatGPT, Claude-4 Sonnet, Claude-3.5 Sonnet to assist with our tasks. While these tools have augmented our capabilities and contributed to our findings, it's pertinent to note that they have inherent limitations. We have made every effort to use AI in a transparent and responsible manner. Any conclusions drawn are a result of combined human and machine insights.

This is an automatic report generated with AI Usage Cards. https://ai-cards.org

# References 

1) Paraphrase Generation with Deep Reinforcement Learning [Li, Jiang, Shang et al., 2018]

2) Quality Controlled Paraphrase Generation, [Bandel et al., 2022]


Write down all your references (other repositories, papers, etc.) that you used for your project.



