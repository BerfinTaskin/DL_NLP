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

### BART Generation
Running the training script for the best improvement for the BART Generation task can be done as follows:
python3 finetune_bart_generation.py --use_gpu

### BART Detection
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM

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

LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM

### BART Generation

We improved the **BART paraphrase generation** baseline along four axes: data handling, optimization, decoding, and evaluation.

**Data handling**
- The finetuned system reuses a single tokenizer across the pipeline and **pre-tokenizes targets** once in the dataloader, avoiding per-batch retokenization.
- The dataloader is configured with CUDA-friendly options (e.g., `pin_memory`, worker processes), which reduces host–device transfer overhead.
- The code also supports **optional control tokens** (e.g., `<SEP>`, `<TYPE_k>`) that can encode paraphrase constraints (span or type) directly in the input sequence, enabling conditional generation when such metadata is available.

**Optimization and training stability**
- The training loop replaces plain SGD-like updates with **AdamW** (decoupled weight decay), couples it with a **linear warmup + linear decay** learning-rate schedule, and adds **gradient clipping**.
- To increase the effective batch size under limited GPU memory, the loop employs **gradient accumulation**, where the effective batch is \( B_{\text{eff}} = B \times \text{accum} \).
- Training uses **mixed-precision (AMP)** with dynamic loss scaling, which both accelerates throughput and reduces numerical instabilities on GPU.
- Finally, the model is validated after each epoch on a held-out development set, with **early stopping** and **best-checkpoint selection**; this prevents overfitting and ensures that downstream evaluation uses the strongest model rather than the last epoch.

**Decoding strategy**
- Inference uses beam search with **n-gram blocking** (`no_repeat_ngram_size`) and a tunable **length penalty**.
- These constraints are known to reduce degeneracies such as verbatim copying and short, repetitive outputs—critical for paraphrase tasks where surface diversity is desirable.

**Evaluation metric (correctness and copy penalty)**
The baseline originally computed SacreBLEU with reversed arguments. We fix the orientation and report:

- `BLEU_ref = BLEU(predictions, references)`
- `BLEU_inp = BLEU(predictions, inputs)`

To explicitly discourage copying from the input, we use the penalized BLEU:

- `pBLEU = BLEU_ref * (100 - BLEU_inp) / 52`

Here, `100 - BLEU_inp` acts as a “diversity” factor and the constant 52 rescales to a 0–100 range (as defined in the assignment). This metric rewards outputs that are both faithful to references and **lexically distinct** from the inputs.

**Empirical Outcome**

Under identical data and evaluation conditions, the finetuned system improves **penalized BLEU** from **8.95** (baseline) to **19.59**. The gain is driven primarily by reduced input copying (lower \( \text{BLEU}_{\text{inp}} \)), with a modest trade-off in raw BLEU vs. references—consistent with the objective of producing paraphrases rather than near-copies.


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
- For LORA, the experiment included 5 different configurations for the rank `lora_r`, the scaling factor `lora_alpha` and `lora_dropout`. The results for the best one are reported in the results section. These five configurations were:

| Config | `lora_r` | `lora_alpha` | `lora_dropout` |
|-------:|:--------:|:------------:|:--------------:|
|   1    |    4     |      16      |      0.00      |
|   2    |    8     |      16      |      0.05      |
|   3    |    8     |      32      |      0.10      |
|   4    |   16     |      32      |      0.10      |
|   5    |   16     |      48      |      0.10      |

### BERT Paraphrase Detection
LOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUMLOREMIPSUM
### BART Generation
We compare a **baseline BART-large** paraphrase generator against an **improved finetuned variant** on the ETPC dataset (train/dev/test).

**Evaluation**
- `BLEU_ref = BLEU(predictions, references)`
- `BLEU_inp = BLEU(predictions, inputs)` (measures similarity to the source; lower means less copying)
- **Penalized BLEU (assignment metric):**

- We expected:
- A more stable optimization recipe (AdamW + warmup/decay, gradient clipping/accumulation, AMP) and **early stopping** should improve generalization.
- **Decoding constraints** (n-gram blocking, length penalty) should **reduce copying**, increasing `100 - BLEU_inp`, and thus improve `pBLEU`, even if `BLEU_ref` drops slightly.

**Per-epoch trend (finetuned):**

| Epoch | BLEU vs refs | BLEU vs inputs | pBLEU |
|------:|--------------|----------------|------:|
| 1     | 48.84        | 95.53          | 4.20  |
| 2     | 48.49        | 90.96          | 8.43  |
| 3     | 48.10        | 88.19          | 10.92 |
| 4     | 46.99        | 84.50          | 14.00 |
| 5     | 46.66        | 82.43          | 15.77 |
| 6     | 44.88        | 77.30          | 19.59 |

- The finetuned system achieves a **+10.64 pBLEU** gain (≈ **2.2×** improvement) over the baseline.
- The improvement is driven primarily by **reduced copying** (lower `BLEU_inp` → higher `100 - BLEU_inp`), which the metric explicitly rewards.
- A small decrease in `BLEU_ref` is expected: discouraging copying can reduce n-gram overlap with references slightly, but the combined objective (pBLEU) improves substantially.
- **Early stopping** selects the best trade-off epoch; the per-epoch table shows `pBLEU` increasing as copying decreases.
- Optimization upgrades (warmup/decay, clipping, AMP) improve **stability and efficiency**, enabling better convergence; decoding constraints then **shape** outputs away from verbatim reuse.
- Overall, results **match expectations** and reveal a clear trend: enforcing diversity during decoding and stability during training yields better paraphrases under the assignment’s metric.

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

| **Semantic Textual Similarity (STS)** | **Correlation** | **Interpretation** |
|---------------------------------------|-----------------|--------------------------------|
| Baseline                              | 0.365           |     baseline         |        
| LORA                         |         0.347   |   much more efficient (tradeoff)         |
| Data augmentations                         |0.350           |  ineffective        |
| Negative Pearson Loss                         | 0.334          |  ineffective   |
| Mean-pooling embedding                         |0.335           |    ineffective     |
| Cosine similarity                         | 0.718          |   better objective   |


| **Paraphrase Type Detection (PTD)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |45.23%           |...            | 
|Improvement 1          |58.56%            |...          
|Improvement 2        |52.11%|...|
|...        |...|...|

| **Paraphrase Type Generation (PTG)** | **BLEU Score** | **Negative BLEU (with input)** | **Penalized BLEU** |
|---|---:|---:|---:|
| Baseline        | 48.420 | 9.610 | 8.950 |
| Improvement 1   | 44.880 | 22.700 | 19.590 |
| Improvement 2   | …      | …     | …     |


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
.....
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



