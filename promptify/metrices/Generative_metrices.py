"""
    This class is to implement the Generative content evaluation functions, to measure how good the content is generated from gpt3, chatgpt or any other generative model
    @author: Aaditya(Ankit) <aadityaura@gmail.com>
    @date created: 27/06/2022
    @date last modified: 02/08/2022
"""

import json
import pandas as pd
from tqdm import tqdm
from textacy import text_stats

class Generation_metrices(object):
    
    def __init__(self, spacy_model):
        self.en  = textacy.load_spacy_lang(spacy_model, disable=("parser",))
        
    def get_score(self, generated_content):
        doc    = textacy.make_spacy_doc(generated_content, lang=self.en)
        
        result = {'Words'      : [text_stats.basics.n_words(doc)], 'Syllables' : [text_stats.basics.n_syllables(doc)],
                  'Entropy'    : [text_stats.basics.entropy(doc)], 'ARI'       : [text_stats.readability.automated_readability_index(doc)],
                  'Flesch KGL' : [text_stats.readability.flesch_kincaid_grade_level(doc)],
                  'Smog Index' : [text_stats.readability.smog_index(doc)]}
        
        
        df = pd.DataFrame(result)
        return df
    
    def plot_data(self, score_df):
        
        """score_df : pandas dataframe"""
        score_df.plot(kind='bar')
        
        
    def bulk_evaluate(self, content_list):
        
        """content_list = ["sentence_1", "sentence_2", "sentence_3"]"""
        
        all_frames = []
        for each_sentence in content_list:
            all_frames.append(self. get_score(each_sentence))
        
        return pd.concat(all_frames).reset_index(drop=True)
