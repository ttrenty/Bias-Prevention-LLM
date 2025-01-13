# LLM Bias Prevention  
**Authors: Sybille Lafont, Thomas Trenty, Caroline Apel** 

---


This project explores methods to identify typical biases that emerge in Large Language Models (LLMs) and applies various strategies to mitigate them. The impact of these mitigation techniques is evaluated both in terms of their effectiveness in reducing bias and their potential side effects by comparing the modified model's performance against the original.  

The definitions of **bias** provided by the Oxford and Cambridge dictionaries are as follows:  
- **Oxford:** A strong feeling in favor of or against one group of people or one side in an argument, often not based on fair judgment.  
- **Cambridge:** The action of supporting or opposing a particular person or thing in an unfair way, because of allowing personal opinions to influence your judgment.  

Determining which ideas or judgments constitute bias is a complex and nuanced task as it involves subjective decisions. Also, selectively removing certain biases while neglecting others—whether inadvertently or intentionally—could potentially introduce further bias into the model. For a deeper discussion on this topic, we recommend OpenAI’s article on their approach to AI behavior where they explain how a group of people can help remove biases in a model: [How should AI systems behave, and who should decide?](https://openai.com/index/how-should-ai-systems-behave/).  

This project does not aim to remove specific biases but rather proposes methods to detect and mitigate a given bias and/or biases in general. The examples provided are illustrative and not intended to reflect the authors’ personal views; they serve to demonstrate the approaches in practice.

## File Organisation

```
.
├── README.md        # The current file
├── LICENSE          # MIT License
├── transformers.yml # Conda Dependencies, use `conda env create -f transformers.yml`
├── hands_on         # Experimental files to showcase some code snippets
│    ├── ***.py         # TODO 
│    ├── ***.ipynb      # TODO 
│    └── ...
├── data             # Text datasets
│    ├── raw            # Raw text datasets
│    │    └── ...  
│    └── transformed    # Folder to hold transformed from 'raw' or generated datasets
│         └── .gitignore
└── results          # Generated files from experiements and model evaluations
     └── ...

```
## Bias Detection Approaches

1. **Manual Testing**  
   Detect biases by manually testing for known biases. For example, sexist biases may emerge in an LLM by recommending different jobs based on the gender of the person mentioned in the query.

2. **Text Analysis for Training Data (Optional)**  
   Since we know that LLMs reproduce biases present in their training data, we can analyze this training data to identify potential biases that may emerge in the LLM after training.  
   The idea is to analyze the texts used for training, or a representative subset, to highlight differences in the treatment of two (or more) ideas. This process can involve constructing a graph of concepts within the texts (using tools like [Neo4j](https://neo4j.com/developer-blog/construct-knowledge-graphs-unstructured-text/), [Text2Graph](https://graphlytic.com/text2graph), or [this blog post](https://www.graphable.ai/blog/text-to-graph-machine-learning/)) and identifying biases arising from unbalanced relationships between multiple given classes (e.g., male and female) and other key ideas.  
   The output of this method would be the identification of key ideas that diverge in their treatment across the multiple input classes in a given set of texts.

## Bias Correction Approaches

For all the different biases one wishes to prevent, the following methods can be used either to address all the biases simultaneously or one bias at a time.

### During Pre-Training:

- **Debiased Contrastive Learning (Optional)**  
  Use a debiased contrastive learning approach (WCL [NOVEL APPROACHES TO MITIGATE DATA BIAS AND MODEL BIAS FOR FAIR MACHINE LEARNING PIPELINES (page 104-107)](https://hammer.purdue.edu/articles/thesis/NOVEL_APPROACHES_TO_MITIGATE_DATA_BIAS_AND_MODEL_BIAS_FOR_FAIR_MACHINE_LEARNING_PIPELINES/25670736?file=45889836)) for fair self-supervised learning. This method introduce the concept of *relative difficulty*, which compares the similarity score with its bias-amplifying counterpart, eliminating the need for annotations of sensitive attributes or target labels.
  We need to make sure this can be applied in the context of LLM training, the data and output being text.

- Mixing pre-training and fine-tuning methods that can act as regulation terms.

### During Fine-Tuning:

1. **Model Self-Regulation**  
   Explicitly instruct the model to avoid producing some identified bias for specific occasions. Either by "Prompt engineering" (prefix tuning, prompt tunining, p-tuning) by providing guidelines to follow or "In context learning" by providing examples to imitate.
   
2. **Supervised fine-tuning (SFT)**

   - **Counter-Bias Fine-Tuning**  
      Fine-tune the model using sentences that counteract the bias (possibly generated by large language models locally to ensure diversity). 
      Target the creation of sentences to the current most biased elements until convergence and iterated until all desired biases have been reduced by a margin within 5 % of each original classes.

   - **Loss Function Adjustment**  
      Modify the loss function during fine-tuning to include a constraint for mitigating the specific bias. This constraint would aim for label distribution outputs within a margin of 5%. Do this constrain on 1 token and then iterate to remove biases on all `n` tokens, or do on all `n` tokens at once.

3. **Direct Preference Optimization (DPO) (Optional)**  
   From https://arxiv.org/abs/2305.18290, more efficient and as effective as Reinforcement learning with human feedback (RLHF).

## Performance Testing

- Evaluate model performance before and after bias mitigation [7, 8, 11] using an unbiased dataset. Ensure that the performance remains consistent [6, 9]. 



## Roadmap

**DONE:**

- Initial approach to evaluate a given bias in an LLM decoder model.

- Preliminary evaluation of ROUGE/BLEU scores for the base and modified models.

- Implementation of a model self-regulation method.
  
- Biais identification related to gender and jobs in Bert predictions tokens

- Starting implementing a model to fine tune bert with a shared loss for male and female, related to jobs

- Implementation of a model to fine tune a text to text architecture to output less stereotyped job assignement.

- Automatics generation of a dataset of sentences related to jobs and gender via LLM request

**Next Steps:**

- Test different biases.

- Evaluate the mitigation of two biases simultaneously and analyze sentences for each bias.

- Generate a bias-free dataset to assess the base model's performance (ROUGE/BLEU).

- Use the TruthfulQA benchmark to evaluate base model performance.

- Perform soft fine-tuning by generating sentences against a specific bias (include saving and loading local models in Transformers).

- Train the T5 architecture on a large dataset  (text to text)

- Finish implementing the model to fine tune Bert

- LLM Judge avec LM Studio

- Encoder --> Decoder : Check if a classifier can classify sentences that are biased in the --> section

- Update BERT to ModernBERT

## Bibliography

Main readings:  

1. **GPT-4 Technical Report**  
   https://arxiv.org/abs/2303.08774

2. **Towards Detecting Unanticipated Bias in Large Language Models**  
    https://arxiv.org/html/2404.02650v1

3. **Novel Approaches to Mitigate Data Bias and Model Bias for Fair Machine Learning Pipelines**  
   https://hammer.purdue.edu/articles/thesis/NOVEL_APPROACHES_TO_MITIGATE_DATA_BIAS_AND_MODEL_BIAS_FOR_FAIR_MACHINE_LEARNING_PIPELINES/25670736

4. **Algorithms: Bias, Discrimination, and Fairness**  
   https://www.telecom-paris.fr/wp-content-EvDsK19/uploads/2019/02/Algorithmes-Biais-discrimination-equite.pdf

5. **Pretrained and Fine-Tunable Large Language Models (PyTorch + Transformers + (Q)Lora)**    
   https://huggingface.co/docs/transformers/en/model_summary
   https://huggingface.co/docs/diffusers/en/training/lora

6. **ROUGE metric**  
    https://en.wikipedia.org/wiki/ROUGE_(metric)

7. **Measuring Implicit Bias in Explicitly Unbiased Large Language Models**  
    https://arxiv.org/abs/2402.04105

8. **TruthfulQA**  
   A benchmark to evaluate a model’s ability to generate both informative and truthful answers.  
   https://github.com/sylinrl/TruthfulQA

9. **Real Prompt Toxicity**  
   A dataset of toxic prompts used to analyze what the model generates in response.  
   https://realtoxicityprompts.apps.allenai.org/

10. **CrowS-Pairs: A Challenge Dataset for Measuring Social Biases in Masked Language Models**  
    https://arxiv.org/abs/2010.00133  
    https://github.com/nyu-mll/crows-pairs

11 **CIVICS: Building a Dataset for Examining Culturally-Informed Values in Large Language Models**  
   https://arxiv.org/html/2405.13974v1  
   https://huggingface.co/CIVICS-dataset

Others:  

12. **Understanding and Mitigating Bias in Large Language Models (LLMs)**  
   https://www.datacamp.com/blog/understanding-and-mitigating-bias-in-large-language-models-llms?dc_referrer=https%3A%2F%2Fwww.google.com%2F

13. **Decoding Biases: Automated Methods and LLM Judges for Gender Bias Detection in Language Models**  
    https://arxiv.org/abs/2408.03907

14. **Large Language Models Are Biased. Can Logic Help Save Them?**  
   https://www.csail.mit.edu/news/large-language-models-are-biased-can-logic-help-save-them

15. **PerspectiveAPI**  
   A tool to measure the toxicity of the model’s generated outputs.  
   https://www.perspectiveapi.com/how-it-works/

16. **Challenging Fairness: A Comprehensive Exploration of Bias in LLM-Based Recommendations**  
    https://arxiv.org/abs/2409.10825v1