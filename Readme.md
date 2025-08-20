# Airline Sentiment Analysis – Internship Assignment

This repository contains my submission for the internship assignment. The tasks focus on **NLP dataset preparation, prompt engineering, model evaluation, and troubleshooting.**


## 1. NLP & Dataset Preparation
- **Dataset Chosen:** [TweetEval – Sentiment Subset](https://huggingface.co/datasets/tweet_eval)  
  - Contains tweets labeled as **positive, negative, or neutral**.  
  - I chose this dataset because it is widely used in benchmarking sentiment analysis tasks and directly fits the problem.  

- **Preprocessing Steps:**  
  - Lowercased text. 
  - Removed URLs, mentions, hashtags, and extra whitespace from the dataset. 
  - I used the Hugging Face pipeline, which automatically handles tokenization and feeding data into the pretrained model process.  

Example of cleaned tweet:  
`"I love the crew, they were very friendly!" -> "i love the crew they were very friendly"`


## 2. Prompt Engineering & Model Interaction
I have designed **three different prompt styles** for sentiment classification:  

1. **Direct Prompt**  
   - Simple instruction asking for sentiment classification.  
   - Works well but sometimes lacks context thus giving incorrect answers.  

2. **Few-Shot Prompt**  
   - Provided labeled examples before the target tweet.  
   - Helps guide the model with reference patterns.  

3. **Chain-of-Thought Prompt**  
   - Asked the model to reason step by step before giving a label.  
   - Useful for ambiguous tweets like the last example used in the tast1_2.ipynb.  


## 3. Model Evaluation
- **Model Used:** `distilbert-base-uncased-finetuned-sst-2-english` (Hugging Face pipeline)  
- **Test Data:** 200 random tweets from the TweetEval dataset.  

**Results (scikit-learn metrics):**  
- Accuracy: ~0.75  
- Precision: ~0.6879595588235294  
- Recall: ~0.7125779625779626  
- F1: ~0.6964545896066052  

These results show that the pretrained model performs well without fine-tuning.


## 4. Troubleshooting
**Issue:** Prompt design can cause **bias or misclassification** (e.g., sarcastic tweets misclassified as positive).  
**Solution:**  
- Use few-shot prompts with diverse examples.  
- If resources allow, fine-tune the model on airline specific tweets for domain adaptation.  

## Conclusion

In this project, we compared two different approaches to sentiment classification:  

- **DistilBERT (fine tuned specialist)** – Achieved strong results (~75% accuracy). Since it is specifically fine tuned on sentiment tasks (SST-2 dataset), it consistently provided reliable predictions.  

- **FLAN-T5 (general purpose instruction model)** – Performance was lower (~40% accuracy), as expected. Unlike DistilBERT, FLAN is not fine tuned for sentiment classification. Rather, it is an instruction following model that can fit a large range of NLP tasks with different prompts.  

**Key Insight:**  
- DistilBERT is like a **specialist**: very accurate when applied to the exact task it was trained for.  
- FLAN-T5 is like a **generalist**: it can handle many tasks (Q&A, summarization, reasoning) without fine tuning, but at the cost of lower accuracy on narrow tasks like sentiment classification.  

This comparison highlights a tradeoff in modern NLP:  
- Use **fine tuned models** when you need the highest accuracy for a specific task.  
- Use **instruction tuned models** (like FLAN or GPT) when you need flexibility across tasks or don’t have task-specific fine-tuned models available.  


## Resource Constraints & Compromises

- **No Fine Tuning:**  
  Due to limited compute, we did not fine tune models like DistilBERT or FLAN T5 on the tweet dataset. Instead, we directly used pretrained versions. Fine tuning would likely have improved accuracy, especially for FLAN T5.  

- **Small Sample Size for Evaluation:**  
  Instead of evaluating on the full dataset, we tested on a **smaller subset** (e.g. 200 samples) to reduce runtime and memory usage.  

- **CPU-only Execution:**  
  Models were run on CPU instead of GPU, which made generation slower and limited experimentation with larger models.  

- **Limited Model Choice:**  
  We used **DistilBERT (sentiment finetuned)** and **FLAN-T5-small**. Larger models (e.g., BERT-large, FLAN-T5-large) could give better accuracy but were too heavy for local execution.  

- **Prompt-Based Evaluation Only:**  
  For FLAN-T5, we relied solely on **prompt engineering** instead of training. This limited performance since the model is not specialized for sentiment classification.  
