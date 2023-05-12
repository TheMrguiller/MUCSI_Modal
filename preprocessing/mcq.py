
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback
from similarity.normalized_levenshtein import NormalizedLevenshtein
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import OrderedDict
import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
from sense2vec import Sense2Vec
from sentence_transformers import SentenceTransformer
import random
from random_word import RandomWords

class MCQ():

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.normalized_levenshtein = NormalizedLevenshtein()
        self.question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
        self.question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
        self.question_model = self.question_model.to(self.device)
        self.s2v = Sense2Vec().from_disk('s2v_old')
    
        self.sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')
        
        self.r = RandomWords()

    def get_nouns_multipartite(self,content):#Gets the keyword from a text
        out=[]
        try:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=content,language='en')
            #    not contain punctuation marks or stopwords as candidates.
            pos = {'PROPN','NOUN'}
            #pos = {'PROPN','NOUN'}
            stoplist = list(string.punctuation)
            stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
            stoplist += stopwords.words('english')
            # extractor.candidate_selection(pos=pos, stoplist=stoplist)
            extractor.candidate_selection(pos=pos)
            # 4. build the Multipartite graph and rank candidates using random walk,
            #    alpha controls the weight adjustment mechanism, see TopicRank for
            #    threshold/method parameters.
            extractor.candidate_weighting(alpha=1.1,
                                        threshold=0.75,
                                        method='average')
            keyphrases = extractor.get_n_best(n=15)
            

            for val in keyphrases:
                out.append(val[0])
        except:
            out = []
            traceback.print_exc()

        return out
    def get_keywords(self,originaltext):#Returns the keywords
        keywords = self.get_nouns_multipartite(originaltext)
        
        important_keywords =[]
        for keyword in keywords:
            important_keywords.append(keyword)

        return important_keywords[:4]
    def get_question(self,context,answer,model,tokenizer):#Obtain question from text and answer
        text = "context: {} answer: {}".format(context,answer)
        encoding = tokenizer.encode_plus(text,max_length=384, pad_to_max_length=False,truncation=True, return_tensors="pt").to(self.device)
        input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

        outs = model.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        early_stopping=True,
                                        num_beams=5,
                                        num_return_sequences=1,
                                        no_repeat_ngram_size=2,
                                        max_length=72)


        dec = [tokenizer.decode(ids,skip_special_tokens=True) for ids in outs]


        Question = dec[0].replace("question:","")
        Question= Question.strip()
        return Question
    def get_distractors (self,word,origsentence,sense2vecmodel,sentencemodel,top_n,lambdaval):
        distractors = self.sense2vec_get_words(word,sense2vecmodel,top_n,origsentence)
        # print ("distractors ",distractors)
        if len(distractors) ==0:
            return distractors
        distractors_new = [word.capitalize()]
        distractors_new.extend(distractors)
        

        embedding_sentence = origsentence+ " "+word.capitalize()
        
        keyword_embedding = sentencemodel.encode([embedding_sentence])
        distractor_embeddings = sentencemodel.encode(distractors_new)

        
        max_keywords = min(len(distractors_new),5)
        filtered_keywords = self.mmr(keyword_embedding, distractor_embeddings,distractors_new,max_keywords,lambdaval)
        # filtered_keywords = filtered_keywords[1:]
        final = [word.capitalize()]
        for wrd in filtered_keywords:
            if wrd.lower() !=word.lower():
                final.append(wrd.capitalize())
        final = final[1:]
        return final
    def mmr(self,doc_embedding, word_embeddings, words, top_n, lambda_param):

        # Extract similarity within words, and between words and the document
        word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
        word_similarity = cosine_similarity(word_embeddings)

        # Initialize candidates and already choose best keyword/keyphrase
        keywords_idx = [np.argmax(word_doc_similarity)]
        candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

        for _ in range(top_n - 1):
            # Extract similarities within candidates and
            # between candidates and selected keywords/phrases
            candidate_similarities = word_doc_similarity[candidates_idx, :]
            target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

            # Calculate MMR
            mmr = (lambda_param) * candidate_similarities - (1-lambda_param) * target_similarities.reshape(-1, 1)
            mmr_idx = candidates_idx[np.argmax(mmr)]

            # Update keywords & candidates
            keywords_idx.append(mmr_idx)
            candidates_idx.remove(mmr_idx)

        return [words[idx] for idx in keywords_idx]
    def sense2vec_get_words(self,word,s2v,topn,question):
        output = []
        # print ("word ",word)
        try:
            sense = s2v.get_best_sense(word, senses= ["NOUN", "PERSON","PRODUCT","LOC","ORG","EVENT","NORP","WORK OF ART","FAC","GPE","NUM","FACILITY"])
            most_similar = s2v.most_similar(sense, n=topn)
            # print (most_similar)
            output = self.filter_same_sense_words(sense,most_similar)
            # print ("Similar ",output)
        except:
            output =[]

        threshold = 0.6
        final=[word]
        checklist =question.split()
        for x in output:
            if self.get_highest_similarity_score(final,x)<threshold and x not in final and x not in checklist:
                final.append(x)
        
        return final[1:]
    def filter_same_sense_words(self,original,wordlist):
        filtered_words=[]
        base_sense =original.split('|')[1] 
        # print (base_sense)
        for eachword in wordlist:
            if eachword[0].split('|')[1] == base_sense:
                filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
        return filtered_words

    def get_highest_similarity_score(self,wordlist,wrd):
        score=[]
        for each in wordlist:
            score.append(self.normalized_levenshtein.similarity(each.lower(),wrd.lower()))
        return max(score)
    def generate_MCQ(self,context):
        dictonary = {}
        np_value =  self.get_keywords(context)
        candidate_2_dict={}
        candidate_1_dict={}

        for idx,answer in enumerate(np_value):
            ques = self.get_question(context,answer,self.question_model,self.question_tokenizer)
            distractors = self.get_distractors(answer.capitalize(),ques,self.s2v,self.sentence_transformer_model,40,0.2)
            if len(distractors)>0:
                distractors = distractors[:2]
            
            dictonary[idx]={"answer":answer,"question":ques,"options":distractors}
            if len(distractors)==2:
                candidate_2_dict[idx]={"answer":answer,"question":ques,"options":distractors}
            elif len(distractors)>0:
                candidate_1_dict[idx]={"answer":answer,"question":ques,"options":distractors}
        # print(dictonary)
        # candidate_1_dict[idx]={"answer":"pepe","question":"question","options":["distr"]}
        if len(candidate_2_dict.keys())>0:
            return candidate_2_dict[random.choice(list(candidate_2_dict.keys()))]
        if len(candidate_1_dict.keys())>0:
            candidate_1=candidate_1_dict[random.choice(list(candidate_1_dict.keys()))]
            candidate_1["options"].extend([self.r.get_random_word()])
            return candidate_1
        if dictonary:
            _,return_value = random.choice(list(dictonary.items()))
            
            np_value.remove(return_value["answer"])
            
            if len(np_value)>=2:
                return_value["options"]=random.sample(np_value, 2)
            else:
                return_value["options"].extend([self.r.get_random_word(),self.r.get_random_word()])
                return_value["options"]=random.sample(return_value["options"], 2)

            return return_value
        else:
            return {"answer":answer,"question":"","options":["",""]}
            
            

