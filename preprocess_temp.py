import pandas as pd
import re
import spacy_stanza
from spacytextblob.spacytextblob import SpacyTextBlob

class preprocess:
    
    def __init__(self, sentence):
        self.sentence = sentence
        self.catNN = ['NN', 'NNS', 'NNP', 'NNPS']#+['JJ', 'JR', 'RB']
        self.nlp = spacy_stanza.load_pipeline("en")
        self.nlp.add_pipe('spacytextblob')
        self.text = self.deEmojify()
        self.doc = self.nlp(self.text)
        self.result = self.extract_aspects()
        self.values = pd.DataFrame(self.result.values())
        if len(self.values):
            self.values['assessment_text'] = self.values['assessment_text'].apply(lambda x: self.joiner(x))
            self.values['mod_text'] = self.values['assessment_text'] + ' ' + self.values['target_text']
            self.mod_list = list(self.values['mod_text'])
        else:
            self.mod_list = self.text.split('.')
        self.phrase_list = []
        for str_ in self.mod_list:
            for str__ in str_.split(' ')[:-1]:
                self.phrase_list.append(' '.join([str__, str_.split(' ')[-1]]))
    def process_spacy(self):
        words, targets, all_targets = [], {}, []
        for word in self.doc:
            if word.tag_ in self.catNN:
                targets[word.text] = [word.tag_, word.lemma_]
                all_targets.append([word.text, word.tag_, word.lemma_])
            words.append(word.text)

        edges = []
        for token in self.doc:
            token_info = [token.text, token.lemma_, (token.is_stop and token.text not in ['no', 'not']), token.pos_, token.tag_]
            token_info = [token.text, token.lemma_, False, token.pos_, token.tag_]
            for child in token.children:
                child_info = [child.text, child.lemma_, (child.is_stop and child.text not in ['no', 'not']), child.pos_, child.tag_]
                child_info = [child.text, child.lemma_, False, child.pos_, child.tag_]
                edges.append([child.dep_, token_info, child_info])

        return words, targets, edges        
        
    def extract_aspects(self):
        words, targets, edges = self.process_spacy()

        sentiment_terms = {}
        ###for item in self.doc._.assessments:
        ###    sentiment_terms[item[0][-1]] = [item[1], ' '.join(item[0])]
        ###print(sentiment_terms)
        targets_assessments = {}
        for target in targets:
            if targets[target][0] not in self.catNN:
                continue
            target_text = targets[target][1]
            assessments = []
            for edge in edges:
                if edge[0] in ['det', 'case', 'root', 'acl:relcl'] and 'no' not in edge[1]+edge[2]:
                #if not dependency[2] in ["nsubj", "acl", "recl", "acl:relcl", "obj", "obl", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"]:
                #if not dependency[2] in ["acl", "recl", "obj", "obl", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"]:
                #if not dependency[2] in ["nsubj", "acl:relcl", "obj", "obl", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"]:
                    continue
                if edge[1][0] == target and edge[2][2] is False:# and dependency[1] not in targets:#
                    if edge[2][0] in targets:# and dependency[2] in ['conj', 'compound', 'nmod']:
                        target_text = target_text + ' ' + edge[2][1]
                    else:
                        assessments.append(edge[2][0])
                elif edge[2][0] == target and edge[1][2] is False:# and dependency[0] not in targets:#
                    if edge[1][0] in targets:# and dependency[2] in ['conj', 'compound', 'nmod']:
                        target_text = edge[1][1] + ' ' + target_text
                    else:
                        assessments.append(edge[1][0])

            if len(assessments) == 0:
                continue

            targets_assessments[target] = dict()
            targets_assessments[target]['target_text'] = target_text
            targets_assessments[target]['assessment_text'] = assessments
            ###sent_terms = [term for term in assessments if term in sentiment_terms]
            ###if len(sent_terms) == 0:
            ###    targets_assessments[target]['sentiment'] = 'unknown'
            ###    targets_assessments[target]['sentiment_text'] = []
            ###elif sentiment_terms[sent_terms[0]][0] >= 0.33:
            ###    targets_assessments[target]['sentiment'] = 'positive'
            ###    targets_assessments[target]['sentiment_text'] = [sentiment_terms[term][1] for term in sent_terms]
            ###elif sentiment_terms[sent_terms[0]][0] <= -0.33:
            ###    targets_assessments[target]['sentiment'] = 'negative'
            ###    targets_assessments[target]['sentiment_text'] = [sentiment_terms[term][1] for term in sent_terms]
            ###else:
            ###    targets_assessments[target]['sentiment'] = 'mixed'
            ###    targets_assessments[target]['sentiment_text'] = [sentiment_terms[term][1] for term in sent_terms]

        return targets_assessments
    
    def deEmojify(self):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',self.sentence)

    def joiner(self, x):
        x = [x_ for x_ in x if x_ != '.']
        return ' '.join(x)
    
    def sent_finder(self):
        return self.nlp(self.text)._.polarity
    
        

