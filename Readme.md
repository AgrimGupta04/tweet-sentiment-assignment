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


