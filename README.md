# NLU Coursework for the task of Evidence Detection

The aim of this project was to develop two automated deep learning pairwise classification models: BiLSTM and BERT.

## BiLSTM

Based on the paper [BiLSTM-Attention and LSTM-Attention via Soft Voting in Emotion Classification](https://www.researchgate.net/publication/329512919_NLP_at_IEST_2018_BiLSTM-Attention_and_LSTM-Attention_via_Soft_Voting_in_Emotion_Classification), we use attention mechanism along with BiLSTM to achieve our aim.

### Description

All the necessary libraries are imported such as keras, sklearn, numpy and pandas. The input data is tokenised using keras to split the input text into tokens. After tokenization, the sequences are padded to ensure they are of the same length. The input layer is then converted to high-dimensional vector representation by passing it to an embedding layer.
After the preprocessing, we pass our data to the model to train.
The trained model is then used to test the performance on the dev dataset. An accuracy of 81.6% was achieved.

### Training and Evaluation file

The training and evaluation code is present in blackboard in the file: `training_evaluation_bilstm.ipynb`. 
### Steps to run the demo file:

1. Download the demo file (demo_bilstm) from blackboard.
2. Download the model weights and the required files present [here](https://drive.google.com/drive/folders/1FM9nxg73cxLtJaDBOZF8-fWZadDx5x-P).
3. The files present would be:

- lstm_model.h5
- train.csv
- test.csv

4. Upload all those files into the colab notebook.
5. Go to Runtime -> Run all.


### Model Card
Model card can be found both in blackboard(model_card_bilstm.md) or on [Hugging Face](https://huggingface.co/RibhavOjha/bilstm-evidence-detection).

## BERT

### Pre-trained model used:

`bert-base-uncased` has been used as the pretrained model. It refers to a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model provided by the Hugging Face Transformers library. It is trained on uncased English text, which means it does not differentiate between lowercase and uppercase letters. This means, for example, the word "Hello" will be treated the same as "hello".

### Description

First, all the necessary libraries are imported such as pytorch, huggingface transformer, sklearn and pandas.
Then, the data is loaded from the csv file. The data is made compatible with what BERT expects by making a custom dataset class.
Just fine-tuning the model without modifications yielded an accuracy of 86% on the dev dataset.
To improve the performance, we added a dropout and a linear classification layer on top of the existing BERT model. This helped in reducing
the overfitting of the data. This helped to improve the model accuracy to 88% on the dev dataset.
We use Adam optimizer and CrossEntropyLoss as our loss function.

### Training and Evaluation file

The training and evaluation code is present in blackboard in the file: `training_evaluation_bert.ipynb`. 

### Steps to run the demo file:

1. Download the demo file (demo_bert.ipynb) from blackboard. 
2. Download the model weights and the required files present [here](https://drive.google.com/drive/folders/10xRR8ZIzW5UYbpkprOp4u8_vkZMirgYk).
3. The files present would be:

- bert_model.pth
- train.csv
- test.csv

4. Upload all those files into the colab notebook.
5. Go to Runtime -> Run all

### Model Card 
Model card can be found both in blackboard(model_card_bert.md) or on [Hugging Face](https://huggingface.co/RibhavOjha/bert-evidence-detection). 
