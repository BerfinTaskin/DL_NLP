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
To make use of the functionality provided in this repository, you first have to set up the environment.
We recommend using Conda. If Conda is installed and activated, run the following command from the repository root:

```
source _setup/setup.sh
```

The baseline results can be obtained as described by using the standard `multitask_classifier.py` file using the default command line interface that was already implemented. The improvements are implemented in separate .py files. For example, the improvements for the STS task are implemented in `improvements_sts.py`. The corresponding experiment scripts and code of this improvement can be found accordingly in the `improvements_sts` folder. Within this folder there is a separate readme that explains how to run the experiments. The same holds for the other improvements. 

# Best improvements
### BERT sentiment analysis
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
### BERT semantic textual similarity:
Running the training script for the best improvement for the STS task can be done as follows:
```
python improvements_sts.py \
    --epochs 8 \
    --option finetune \
    --use_gpu \
    --seed 50 \
    --augment_prob 0.0 \
    --head_style 2 \
    --embedding_style cls \
    --loss_function mse \
    --hidden_dropout_prob 0.3 \
    --lr_backbone 1e-4 \
    --lr_head 1e-4 \
    --batch_size 128 \
```
### BERT Paraphrase Detection
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
### BART generation
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
### BART Detection
Running the training script for the best improvement for the paraphrase detection task can be done as follows:

# Methodology
### BERT sentiment analysis
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
### BERT Semantic Textual Similarity (STS):
We aimed to improve upon the baseline using several small measures that are proven to often benefit model performance.

1. **LoRA**: We applied Low-Rank Adaptation (LoRA) to the query, key and value modules of the transformer layers of the BERT model. LoRA introduces trainable low-rank matrices into these layers of the transformer architecture, allowing for efficient fine-tuning with a reduced number of parameters. This method helps in adapting large pre-trained models to specific tasks without extensive computational resources. The smaller number of parameters also helps reduce overfitting, which might be useful in our case since the STS dataset is relatively small.

2. **Data augmentations**: Because the STS dataset is relatively small, we also investigated stochastic text augmentations during training (controlled by `augment_prob`). For each sentence, up to three operations may be applied based on `augment_prob`:

   - **Synonym replacement**: Replaces up to `n=1` non-stopwords with a synonym from WordNet that matches the POS tag. Synonyms are filtered to exclude multi-word phrases, non-alphabetic tokens, and words longer than 15 characters. We also require a Zipf frequency ≥ 2.5 to avoid rare or unnatural words. Simple rules adjust morphology (e.g., plural endings, `-ing`, `-ed`) and capitalization to match the original word.  
   - **Random deletion**: Each token is dropped with probability `p=0.1`. If all tokens would be removed, one random word is retained.  
   - **Random swap**: Swaps the positions of two random words, repeated `n=1` time per augmentation call.

   These perturbations might help increase both lexical and structural diversity while keeping sentences semantically close to the original.

3. **Negative Pearson Loss**: In the baseline, MSE was used to measure the deviation between predicted and ground truth similarity scores. However, since STS evaluation is based on the Pearson correlation coefficient, it might be beneficial to use a loss function that is directly aligned with this metric. We therefore implemented **negative Pearson correlation** as an alternative loss function. Note that this loss requires a batch size greater than 1, as correlation is undefined for a single sample. The Pearson correlation coefficient \(r\) between the predicted scores \(pred\) and the gold scores \(gold\) is computed as:

   \[
   r = \frac{\sum (pred_i - \bar{pred})(gold_i - \bar{gold})}{\sqrt{\sum (pred_i - \bar{pred})^2}\;\sqrt{\sum (gold_i - \bar{gold})^2}}
   \]

   The training loss is then defined as \(1 - r\), so that maximizing correlation corresponds to minimizing the loss.

4. **Mean-pooling sentence embeddings**: Instead of only relying on the [CLS] token representation, we also implemented a mean-pooling strategy across all token embeddings. Specifically, the final hidden states are averaged across the sequence length while applying the attention mask to ignore padding positions. This produces a mask-aware sentence embedding that might capture semantic information more robustly than a single token representation. The pooled embedding is then used as input to the regression layer.

5. **Cosine similarity between sentence embeddings**: In the baseline implementation, a linear layer is used to process the concatenated [CLS] tokens of the two sentences, followed by a sigmoid layer. Instead, we attempted to first process the sentence embeddings using a linear layer and then directly computing the cosine similarity between these two sentence embeddings. The resulting value is then scaled to the [0,5] range to comply with the task. This approach appears to be more aligned with the nature of the STS task, where the semantic similarity between two sentence pairs is of interest.
BERT Paraphrase Detection

### BERT Paraphrase Detection

We try to implement the baseline performace using following techniques

1. **Make the head return logits** In the baseline model, the classifier head returned probabilities after applying a sigmoid activation. Instead, we modified the head to return raw logits directly. This allows the use of more stable loss functions (such as BCEWithLogitsLoss), avoids premature squashing of values, and lets the loss handle the sigmoid internally for improved numerical stability.

2. **Mean Pool over token embeddings**
Rather than using the hidden state of the first token (similar to a [CLS] representation), we employed mean pooling across all token embeddings in the sentence pair. This representation takes into account the entire sentence context, especially beneficial for BART which does not have a dedicated [CLS] token. Mean pooling produced more informative sentence-level features for paraphrase classification.

4. **K-Bin Ensemble**
To reduce variance and improve robustness, we trained an ensemble of models on different random partitions (“bins”) of the training data. Each model was trained independently on one bin, and predictions were aggregated by averaging probabilities across models. This K-bin ensembling helps stabilize performance, reduces sensitivity to random initialization, and improves generalization on the dev and test sets.

We combined techniques such as logits output, BCEWithLogitsLoss with pos_weight, mean pooling, and K-bin ensembling, alongside regularization strategies like gradient clipping, dropout, and learning-rate scheduling. These changes helped stabilize training, handle class imbalance, and improve generalization, with the best results achieved when methods were applied together.

### BART generation

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
### BART Detection
### Hyperparameter Search (Baseline)

To tune hyperparameters, we performed a grid search over 9 parameter combinations, varying the learning rate 
[2e−3,2e−4,2e−5]
[2e−3,2e−4,2e−5] and batch size 
[2,16,32]
[2,16,32]. The results of this sweep are summarized in the table below.


| Learning Rate ↓ / Batch Size → | 2   | 16  | 32  |
|--------------------------------|-----|-----|-----|
| **2e-3**                       |  –  |  –  |  –  |
| **2e-4**                       |  –  |  –  |  –  |
| **2e-5**                       |  –  |  –  |  –  |


# Experiments
### BERT sentiment analysis
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
### BERT semantic textual similarity (STS):
To obtain a strong baseline, we conducted a basic hyperparameter search using the standard implementations of the first part of this module. The hyperparameter search varied the parameters `lr_backbone` (backbone learning rate), `lr_head` (learning rate of small head network), `batch_size`, and `hidden_dropout_prob` around the values that were given by default. We ran one repetition of each configuration such that some noise might have lead to not picking the optimal solution. However, we considered the result good enough. The best configuration found was:
- `lr_backbone`: 0.0001
- `lr_head`: 0.0001
- `batch_size`: 128
- `hidden_dropout_prob`: 0.3

Based on this configuration, we tested the influence of each of the five proposed improvements individually. In each of these five experiments, we kepts the baseline parameters fixed and only activated one of the improvements. 
- For the data augmentation experiment, we set `augment_prob` to 0.25. 
- For LORA, the experiment included 5 different configurations for the rank `lora_r`, the scaling factor `lora_alpha` and `lora_dropout`. These five configurations were:

| Config | `lora_r` | `lora_alpha` | `lora_dropout` |
|-------:|:--------:|:------------:|:--------------:|
|   1    |    4     |      16      |      0.00      |
|   2    |    8     |      16      |      0.05      |
|   3    |    8     |      32      |      0.10      |
|   4    |   16     |      32      |      0.10      |
|   5    |   16     |      48      |      0.10      |

### BERT Paraphrase Detection
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
### BART generation
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
### BART Detection
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM

<!-- Keep track of your experiments here. What are the experiments? Which tasks and models are you considering?

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
- Discuss the results. Why did improvement _A_ perform better/worse compared to other improvements? Did the outcome match your expectations? Can you recognize any trends or patterns? -->

# Results 
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

| **Semantic Textual Similarity (STS)** | **Correlation** |
|---------------------------------------|-----------------|
| Baseline                              | 0.365           |
| LORA                         |            |
| Data augmentations                         |0.350           |
| Negative Pearson Loss                         | 0.334          |
| Mean-pooling embedding                         |0.335           |
| Cosine similarity                         | 0.718          |


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
 
<!-- ### Hyperparameter Optimization 
<!-- Describe briefly how you found your optimal hyperparameter. If you focussed strongly on Hyperparameter Optimization, you can also include it in the Experiment section.  -->

<!-- _Note: Random parameter optimization with no motivation/discussion is not interesting and will be graded accordingly_ --> -->

# Visualizations 
## BERT sentiment analysis
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
## BERT semantic textual similarity (STS):
All experiments showed the best performance very fast within 2-3 epochs, with degrading validation performance from there on. This is in line with the fact that the dataset is relatively small and overfits fast. The training plots were therefore in general not considered interesting. The plot that shows training and validation performance for the best baseline configuration over the 8 training epochs was plotted as an example:
![STS Train vs Dev Correlation](plots/sts_train_vs_dev_corr.png)
## BERT Paraphrase Detection
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
## BART generation
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
## BART Detection
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM

<!-- Add relevant graphs of your experiments here. Those graphs should show relevant metrics (accuracy, validation loss, etc.) during the training. Compare the  different training processes of your improvements in those graphs. 

For example, you could analyze different questions with those plots like: 
- Does improvement A converge faster during training than improvement B? 
- Does Improvement B converge slower but perform better in the end? 
- etc... -->

## Members Contribution 
Explain what member did what in the project:

**Berfin Taskin:** : Worked with bart generation task both for the first part of the project and the second part of the project(improvement), README file, group leading.

**Ashutosh Jaiswal:** :

**Jonathan Henrich:** : Implemented the improvement for the STS baseline and contributed to all components of the BERT-related tasks of the baseline.

**Emre Semercioglu:** :

**Shrinath Madde:** :

...

# AI-Usage Card

In the conduct of this research project, we used specific artificial intelligence tools and algorithms ChatGPT, Claude-4 Sonnet, Claude-3.5 Sonnet to assist with our tasks. While these tools have augmented our capabilities and contributed to our findings, it's pertinent to note that they have inherent limitations. We have made every effort to use AI in a transparent and responsible manner. Any conclusions drawn are a result of combined human and machine insights.

This is an automatic report generated with AI Usage Cards. https://ai-cards.org

# References 

1) Paraphrase Generation with Deep Reinforcement Learning [Li, Jiang, Shang et al., 2018]

2) Quality Controlled Paraphrase Generation, [Bandel et al., 2022]


<!-- Write down all your references (other repositories, papers, etc.) that you used for your project. -->



