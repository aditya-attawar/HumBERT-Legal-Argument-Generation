# humBERT: BERT based legal conversation agent

# Take 1 of JudgeBERT integration with humBERT chassis.

# Given a sentence S, JudgeBERT tells humBERT which case C the argument derives its context from.

# humBERT then tokenizes the sentence S and focuses on the part of the case file where the conversation can possibly originate.

# This is done using sentence_transformers package. We basically convert S into a 768 dimensional embedding and perform 
# cosine similarity comparison to get the sentence which is contextually the closest. Once we know which sentence this is
# we simply pick the 5 lines leading and lagging this sentence to get our focused context, ready to be applied on pretrained BERT for QA/NSP/Sentiment.

# For the sake of saving time we are *not* training the humBERT "brain" on our legal corpus since the two rounds of focusing have made it
# convenient for us to use English LMs directly.

# GPT however will be pre-trained just like BERT (using run_language_modeling.py I think) and will be used as a sentence generator (The Mouth) instead.

# The history unit will fit in somewhere in between the above two steps....


import pandas as pd
import codecs
from nltk.tokenize import sent_tokenize
from pathlib import Path
from transformers import BertTokenizer
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig, BertForQuestionAnswering
from transformers import get_linear_schedule_with_warmup
import numpy as np
import time
import datetime
import random
import os, sys
import heapq
import pickle
from sentence_transformers import SentenceTransformer, models, SentencesDataset, losses
import scipy
import subprocess
from sentence_transformers.readers import NLIDataReader
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import copy
import shutil
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf
import torch.cuda as cutorch


def normalize(x):
    return x/np.linalg.norm(x)


def sentGen(seed, num_sent, model, input_ids):
  tf.random.set_seed(seed)

  # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
  sample_outputs = model.generate(
      input_ids,
      do_sample=True,
      max_length=100,
      top_k=50,
      top_p=0.95,
      num_return_sequences=num_sent
  )

  return sample_outputs

def history_correlation(seed, num_sent, iter, modelSpeech, input_ids, modelVision, tokenizerSpeech, history):
    sample_outputs = sentGen(seed, num_sent, modelSpeech, input_ids)
    if (iter == 0):
        a_rand = np.random.random_integers(3) - 1
        a_randSent = sample_outputs[a_rand]
        return a_randSent
    else:
        cosineSimDistribution = []
        for s in sample_outputs:
            stext = tokenizerSpeech.decode(s, skip_special_tokens=True)
            sample_emb = modelVision.encode([stext])
            history_emb = modelVision.encode(history)
            cosineSimilarity = []
            for h in range(len(history_emb)):
                cosineSimilarity.append(1-scipy.spatial.distance.cosine(history_emb[h], sample_emb[0]))
            cosineSimDistribution.append((max(cosineSimilarity), s))
        cosineSimDistribution.sort(key=lambda x: x[0], reverse=True)
        return cosineSimDistribution[0][1]


# Step 0: Setting which GPU we will be accessing for model loading and calls
device = torch.device("cuda:0")

# Step 1: Getting the case file paths for JudgeBERT to process
# BUGFIX: PathList MUST BE EXPORTED with every model!
# Fortunately we saved this list!


with open ('/data/home/apallaprolu/BERT-Optimized/pathList', 'rb') as fp:
    pathsRefined = pickle.load(fp)

print("INFO: Found a total of " + str(len(pathsRefined)) + " cases...")

# Step 2: Loading pre-trained JudgeBERT from disk (we cannot afford to pre-train every time since RT is over 20 hours)

judgeBERT_Model = BertForSequenceClassification.from_pretrained("/data/home/apallaprolu/BERT-Optimized/model_save/")
judgeBERT_Tokenizer = BertTokenizer.from_pretrained("/data/home/apallaprolu/BERT-Optimized/model_save/")
judgeBERT_Model.to(device)
judgeBERT_loading_done = time.time()
print("INFO: Finished loading pre-trained model from disk.") 

# Step 3: Definition of JudgeBERT: Take the input sentence and classify it into one of the pre-trained labels.

def judgeBERT(tmpSent, tokenizer, model, pathsRefined, device):
    encoded_dict = tokenizer.encode_plus(
                        tmpSent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    tmpOutput = model(encoded_dict['input_ids'].to(device), token_type_ids=encoded_dict['token_type_ids'].to(device), attention_mask=encoded_dict['attention_mask'].to(device))
    logits = tmpOutput[0].detach().cpu().numpy()[0]
    dominating_caseIDs = heapq.nlargest(5, range(len(logits)), logits.take)
    for cid in dominating_caseIDs:
        if cid == 0:
            print(0, "Bad Grammar")
        elif cid == 1:
            print(1, "Correct Grammar but no relevant case found")
        else:
            print(cid, "Case file: " + str(pathsRefined[cid-2]))
    return dominating_caseIDs

# Step 4: Initialization text. This will set the context for the argument as a whole, by picking a relevant subcorpus from the dataset.

test = sys.argv[1] 

print("INFO : Initialization text is : " + test)

# Step 5 : JudgeBERT gets the contextually relevant file from the list of all case files.
# In this case it gets the top 5 relevant files.

caseContexts = judgeBERT(test, judgeBERT_Tokenizer, judgeBERT_Model, pathsRefined, device)
print("INFO : JudgeBERT evaluation is complete.") 


topContext = 0

# Step 5A: If the initialization text is non-legal in form we need a normal text corpus, which we don't have now...
# We could also do a concatenation of top N paths deemed significant by JudgeBERT, and get a wider context....
if caseContexts[0] == 0 or caseContexts[0] == 1:
    print("Normal language text corpus missing, exiting now...")
    # replace this dead-end with topContext = "/path/to/normal/text/corpus"
    sys.exit(0)
else:
    topContext = pathsRefined[caseContexts[0]-2]

print("INFO: JudgeBERT is accessing case file: " + topContext)
# Step 6: Open the top context file if JudgeBERT determines that the initialization is indeed legal in nature.
#         Also tokenize the file into sentences so that humBERT vision module can select the subcontext quickly.
with open(topContext) as f:
    caseSentences = list(map(lambda x: x.strip(), f.readlines()))


# Step 7: humBERT vision module constructs vector embeddings for both the initialization text and the sentences in the case file.
# We *train* the humBERT Vision module on the case text.
# First we create cache directories.
try:
    os.mkdir("./humBERT1Vision/")
    os.mkdir("./humBERT1Vision_save/")
except:
    shutil.rmtree('./humBERT1Vision/')
    shutil.rmtree('./humBERT1Vision_save')
    os.mkdir("./humBERT1Vision/")
    os.mkdir("./humBERT1Vision_save/")

# We then create entailment sequences. Map each sentence in the text to its next sentence.
# This is a very basic model for subcontext derivation and modifications to various mappings can create more complex logical patterns.

s1 = copy.deepcopy(caseSentences)
del(s1[-1])
s2 = caseSentences[1:]
caseLabels = ['entailment']*len(s2)


# Therefore we have arrays s1, s2 where s1[i] -> s2[i] always.
# Let us also load the Stanford NLI corpus as padding to prevent overfitting to legal contexts.

with open("/data/home/apallaprolu/integtration/sentenceTransformerDataSet/AllNLI/s1.golden") as f:
    s1GoldenLines = list(map(lambda x: x.strip(), f.readlines()))
with open("/data/home/apallaprolu/integtration/sentenceTransformerDataSet/AllNLI/s2.golden") as f:
    s2GoldenLines = list(map(lambda x: x.strip(), f.readlines()))
with open("/data/home/apallaprolu/integtration/sentenceTransformerDataSet/AllNLI/labels.golden") as f:
    labelsGoldenLines = list(map(lambda x: x.strip(), f.readlines()))

print("INFO: Finished loading NLI base corpus for vision training...") 

# We will collect all our stuff in one Pandas dataframe and take a random slice.
corpusDF = pd.DataFrame()
corpusDF['s1'] = s1GoldenLines
corpusDF['s2'] = s2GoldenLines
corpusDF['label'] = labelsGoldenLines

# We will also load our case entailment sequences in a dataframe for compatibility.
caseDF = pd.DataFrame()
caseDF['s1'] = s1
caseDF['s2'] = s2
caseDF['label'] = caseLabels

# We oversample our legal corpus twice but dilute it with the Stanford NLI corpus by the same amount for regularization.
caseDF = caseDF.append(caseDF)
dilutionDF = corpusDF.sample(n=2*len(caseDF), random_state=random.randint(1, 1000000))
caseDF = caseDF.append(dilutionDF)

# We save our new *fake* NLI like dataset in our local directory for training.
with open("./humBERT1Vision/s1.train", "w") as f:
    for i in range(len(caseDF['s1'].tolist())):
        f.write(caseDF['s1'].tolist()[i] + "\n")

subprocess.call(["gzip", "./humBERT1Vision/s1.train"])

with open("./humBERT1Vision/s2.train", "w") as f:
    for i in range(len(caseDF['s2'].tolist())):
        f.write(caseDF['s2'].tolist()[i] + "\n")

subprocess.call(["gzip", "./humBERT1Vision/s2.train"])

with open("./humBERT1Vision/labels.train", "w") as f:
    for i in range(len(caseDF['label'].tolist())):
        f.write(caseDF['label'].tolist()[i] + "\n")

subprocess.call(["gzip", "./humBERT1Vision/labels.train"])

print("INFO: Finished hacking NLI corpus with case text....")

# This training is done only once per conversation context setting.
# We use BERT core and build our word embedding generator on top of it!
word_embedding_model = models.Transformer('bert-base-uncased')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode_mean_tokens=True, pooling_mode_cls_token=False, pooling_mode_max_tokens=False)
humBERTVision_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
humBERTVision_model.to(device)
nli_reader = NLIDataReader('./humBERT1Vision/')
num_epochs = 2
train_data = SentencesDataset(nli_reader.get_examples('train.gz'), model=humBERTVision_model)
dev_dataloader = DataLoader(train_data, shuffle=False, batch_size=16)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
train_loss = losses.SoftmaxLoss(model=humBERTVision_model, sentence_embedding_dimension=humBERTVision_model.get_sentence_embedding_dimension(), num_labels=3)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
humBERTVision_model.fit(train_objectives=[(train_dataloader, train_loss)], evaluator=evaluator, epochs=num_epochs, evaluation_steps=1000, output_path="./humBERT1Vision_save/")

print("INFO: HumBERT Vision Training Complete.")

# Once training is done, we can see where our query text correlates the most in the case text.
query_emb = humBERTVision_model.encode([test])
corpus_emb = humBERTVision_model.encode(caseSentences)

cosineSimilarity = []

for i in range(len(corpus_emb)):
    cosineSimilarity.append(1-scipy.spatial.distance.cosine(corpus_emb[i], query_emb[0]))

hotspots = heapq.nlargest(5, range(len(np.array(cosineSimilarity))), np.array(cosineSimilarity).take)

subcontext = ""

for i in hotspots:
    subcontext = subcontext + caseSentences[i]


# We will run language modeling for the specific case that JudgeBERT selects so that
# we have context driven text generation
# This is the humBERT Speech module

try:
    os.mkdir("./humBERT1Speech/")
    os.mkdir("./humBERT1Speech_save/")
except:
    shutil.rmtree('./humBERT1Speech/')
    shutil.rmtree('./humBERT1Speech_save')
    os.mkdir("./humBERT1Speech/")
    os.mkdir("./humBERT1Speech_save/")

torch.cuda.empty_cache()
os.system("python3 run_language_modeling.py --output_dir=./humBERT1Speech_save/ --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=" + str(topContext)+" --do_eval --eval_data_file=" + str(topContext) + " --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 1")

print("INFO: HumBERT Speech Training Complete.") 

humBERTSpeech_tokenizer = GPT2Tokenizer.from_pretrained("./humBERT1Speech_save/")
humBERTSpeech_model = TFGPT2LMHeadModel.from_pretrained("./humBERT1Speech_save/", pad_token_id=humBERTSpeech_tokenizer.eos_token_id, from_pt = "true")

answer = " ".join(list(filter(lambda q: len(q) > 0, caseSentences[hotspots[0]].split(" "))))
input_ids = humBERTSpeech_tokenizer.encode(answer, return_tensors='tf')
history_GPT = []
iter = 0  # GPT conversation number
seed = np.random.randint(3000)
num_sent = 3
history_GPT_new = history_correlation(seed, num_sent, iter, humBERTSpeech_model, input_ids, humBERTVision_model, humBERTSpeech_tokenizer, history_GPT)
history_GPT.append(humBERTSpeech_tokenizer.decode(history_GPT_new, skip_special_tokens=True))

print("\n\n")
print("BEGIN CONVERSATION:\n")
print("====================================\n")
print("Q0: " + str(test) + "\n")
print("A0: " + str(humBERTSpeech_tokenizer.decode(history_GPT_new, skip_special_tokens=True)) + "\n")
response = input(">")
idx = 1
humBERTVision_model.to(torch.device("cuda:"+str(torch.cuda.current_device())))
while(response != "bye"):
    # QUERY -> SUBCONTEXT
    query_emb = humBERTVision_model.encode([response]) 
    cosineSimilarity = []
    for i in range(len(corpus_emb)):
        cosineSimilarity.append(1-scipy.spatial.distance.cosine(corpus_emb[i], query_emb[0]))
    hotspots = heapq.nlargest(5, range(len(np.array(cosineSimilarity))), np.array(cosineSimilarity).take)
    answer = " ".join(list(filter(lambda q: len(q) > 0, caseSentences[hotspots[0]].split(" "))))

    # SUBCONTEXT -> RESPONSE

    history_GPT.append(response)
    history_GPT.append(answer)
    input_ids = humBERTSpeech_tokenizer.encode(answer, return_tensors='tf')
    seed = np.random.randint(3000)
    num_sent = 3
    history_GPT_new = history_correlation(seed, num_sent, iter, humBERTSpeech_model, input_ids, humBERTVision_model, humBERTSpeech_tokenizer, history_GPT) 
    print("Q"+str(idx)+": " + str(response)+"\n")
    print("A"+str(idx)+": " + str(humBERTSpeech_tokenizer.decode(history_GPT_new, skip_special_tokens=True)) + "\n")
    history_GPT.append(humBERTSpeech_tokenizer.decode(history_GPT_new, skip_special_tokens=True)) 
    iter = iter + 3
    if iter % 12 == 0:
        history_GPT = history_GPT[-4:]
    # WAIT FOR NEXT INPUT

    response = input(">")
    idx = idx + 1

