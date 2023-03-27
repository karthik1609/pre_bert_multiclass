import pandas as pd
import re
import spacy_stanza
from spacytextblob.spacytextblob import SpacyTextBlob
import os, sys
from os.path import exists
from fastai.text.all import *
import numpy as np
from os.path import exists
from joblib import dump, load
from sklearn.tree import DecisionTreeClassifier
import fasttext.util

class preprocess:
    
    def __init__(self, sentence, ext_nlp = True, nlp = None):
        self.ext_nlp = ext_nlp
        self.nlp = nlp
        self.sentence = sentence
        self.catNN = ['NN', 'NNS', 'NNP', 'NNPS']#+['JJ', 'JR', 'RB']
        if not ext_nlp:
            self.nlp = spacy_stanza.load_pipeline("en", use_gpu=False)
        self.text = self.deEmojify()
        
    def phrase_extract(self):
        result = self.extract_aspects()
        values = pd.DataFrame(result.values())
        if len(values):
            values['assessment_text'] = values['assessment_text'].apply(lambda x: self.joiner(x))
            values['mod_text'] = values['assessment_text'] + ' ' + values['target_text']
            mod_list = list(values['mod_text'])
        else:
            mod_list = self.text.split('.')
        phrase_list = []
        for str_ in mod_list:
            for str__ in str_.split(' ')[:-1]:
                phrase_list.append(' '.join([str__, str_.split(' ')[-1]]))
        if not phrase_list:
            phrase_list = self.text.split(' ')
        return phrase_list
                    
    def process_spacy(self):
        self.doc = self.nlp(self.text)
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
        self.nlp.add_pipe('spacytextblob')
        return self.nlp(self.text)._.polarity


class vectorize:
    def __init__(self, phrase_list, ext_model = True, model = None):
        self.ext_model = ext_model
        self.d2vmodel = model
        self.phrases = phrase_list
        
    def array(self):
        if not self.ext_model:
            fasttext.util.download_model('en', if_exists='ignore')
            self.d2vmodel = fasttext.load_model('cc.en.300.bin')         
        vec = []
        for phrase in self.phrases:
            for word, order in zip(phrase.split(' '), range(len(phrase.split(' ')))):
                if not order:
                    vec.append(self.d2vmodel.get_word_vector(word))
                else:
                    vec[-1] = np.vstack((vec[-1], self.d2vmodel.get_word_vector(word)))
                if len(phrase.split(' ')) == order + 1:
                    vec[-1] = np.expand_dims(vec[-1], axis=0)

        plarray = np.concatenate(vec, axis=0)
        
        return plarray

class process:
    
    def __init__(self, train_file_string = 'balanced_data.xlsx', test_file_string = 'Yulu_dataanalysis.xlsx', language_file_string = 'yulu.gplay.json'):
        self.train_df = pd.read_excel(train_file_string).fillna('').drop(['Unnamed: 0'], axis = 1)
        self.train_df = self.train_df.reset_index()
        min_train = min(self.train_df['index'])
        self.train_df['index'] = self.train_df['index'].apply(lambda x: x - min_train)
        self.test_df = pd.read_excel(test_file_string).rename(columns = {'Unnamed: 0': 'index'}).fillna('')
        min_test = min(self.test_df['index'])
        self.test_df['index'] = self.test_df['index'].apply(lambda x: x - min_test)
        self.class_list = list(self.train_df.columns[2:])
        self.df_orig = self.test_df.copy()
        self.df = pd.read_json(language_file_string).reset_index().rename(columns = {'content': 'rev_content'})
        def str2num(x):
            if x != '':
                return 1
            else:
                return 0

        for col in self.train_df.columns:
            if col not in ['rev_content', 'index']:
                self.train_df[col] = self.train_df[col].apply(lambda x: str2num(x))
                self.test_df[col] = self.test_df[col].apply(lambda x: str2num(x))
        label_cols = list(self.train_df.columns[2:])

        def get_labels(row):
            #print(row)
            indcs = np.where(row != 0)[0]
            #print(indcs)
            if len(indcs) == 0:
                return "bland"
            return ";".join([label_cols[x] for x in indcs])

        labels = self.train_df[label_cols].apply(lambda row: get_labels(row), axis = 1)
        self.train_df["Labels"] = labels

        labels = self.test_df[label_cols].apply(lambda row: get_labels(row), axis = 1)
        self.test_df["Labels"] = labels
        
    def model_data_prep(self, lang_train = False, class_train = False):
        rets = []
        if lang_train:
            dls_lm = TextDataLoaders.from_df(
                self.df,
                text_col = 'rev_content',
                valid_pct = .2,
                is_lm = True,
                seq_len = 72,
                bs = 64
            )
            rets.append(dls_lm)
        if class_train:
            dls_lm = TextDataLoaders.from_df(
                self.df,
                text_col = 'rev_content',
                valid_pct = .2,
                is_lm = True,
                seq_len = 72,
                bs = 64
            )
            dls_blk = DataBlock(
                blocks = (
                    TextBlock.from_df(
                        text_cols = 'rev_content', 
                        seq_len = 128, 
                        vocab = dls_lm.vocab
                    ),
                    MultiCategoryBlock
                ),
                get_x = ColReader(
                    cols = 'text'
                ),
                get_y = ColReader(
                    cols = 'Labels', 
                    label_delim = ";"
                ),
                splitter = TrainTestSplitter(
                    test_size = 0.2, 
                    random_state = 42
                )
            )

            dls_clf = dls_blk.dataloaders(
                self.train_df,
                bs = 64,
                seed = 42
            )
            rets.append(dls_clf)
        return tuple(rets)
    
    def model_train(self, lang_train = not exists('models/encoder_v1.pth'), class_train = not exists('models/classifier.v5.pth')):

        rets = []
        if not exists('models/encoder_v1.pth'):            
            lang_train = True
        if lang_train and class_train:
            dls_lm, dls_clf = self.model_data_prep(True, True)
        elif lang_train and not class_train:
            dls_lm = self.model_data_prep(True, False)[0]
        elif not lang_train and class_train:
            dls_clf = self.model_data_prep(False, True)[0]
            
        if lang_train:
            learn = language_model_learner(
                dls_lm,
                AWD_LSTM,
                drop_mult = .3,
                
                metrics = [
                    accuracy, 
                    #accuracy_multi,
                    Perplexity()
                ]
            ).to_fp16()
            with learn.no_bar(), learn.no_logging():
                learn_lr = learn.lr_find().valley
                learn.fit_one_cycle(1, learn_lr)
                learn.unfreeze()
                learn.fit_one_cycle(
                    100, 
                    learn_lr, 
                    cbs=[
                        EarlyStoppingCallback(
                            monitor='accuracy', 
                            min_delta=0, 
                            patience=2
                        ), 
                        ReduceLROnPlateau(
                            monitor='valid_loss', 
                            comp=None, 
                            min_delta=0.05, 
                            patience=2, 
                            factor=2.0, 
                            min_lr=0, 
                            reset_on_fit=True
                        )
                    ]
                )
            learn.save_encoder('encoder_v1')
            rets.append(learn)
        if class_train:
            learn_clf = text_classifier_learner(
                dls_clf, 
                AWD_LSTM, 
                drop_mult=0.5,
                metrics = accuracy_multi
            ).to_fp16()
            with learn_clf.no_bar(), learn_clf.no_logging():
                learn_clf = learn_clf.load_encoder('encoder_v1')   
                learn_clf.recorder.silent = True
                learn_clf_lr = learn_clf.lr_find().valley
                learn_clf.fit_one_cycle(1, learn_clf_lr)
                learn_clf.freeze_to(-2)
                learn_clf.fit_one_cycle(1, slice(1e-2/(2.6**4), 1e-2))
                learn_clf.freeze_to(-3)
                learn_clf.fit_one_cycle(1, slice(5e-3/(2.6**4), learn_clf_lr))
                learn_clf.unfreeze()
                learn_clf.fit_one_cycle(
                    100, 
                    slice(1e-3/(2.6**4), learn_clf_lr),
                    #1e-3,
                    cbs=[
                        EarlyStoppingCallback(
                            monitor='accuracy_multi', 
                            min_delta=0.0001, 
                            patience=5
                        )
                    ]
                )
            learn_clf.save('classifier.v5')
            rets.append(learn_clf)            
        return tuple(rets)
    
    def predictor(self, lang_train = False, class_train = not exists('models/classifier.v5.pth')):
        if class_train:
            _, learn_clf = self.model_train(False, True)
        if not class_train:
            dls_clf = self.model_data_prep(False, True)[0]
            learn_clf = text_classifier_learner(
                dls_clf, 
                AWD_LSTM, 
                drop_mult=0.5,
                metrics = accuracy_multi
            ).to_fp16()
            learn_clf = learn_clf.load('classifier.v5')
        return learn_clf
    
    def output_interpreter(self, predictions, create = not exists('models/interpreter.clf')):
        def str2num(x):
            if x == 'Positive':
                return 1
            elif x == 'Negative':
                return -1
            else:
                return 0
        for class_ in self.df_orig.columns[2:]:
            self.df_orig[class_] = self.df_orig[class_].apply(lambda x: str2num(x))
        preds = predictions.copy()
        assessment_list = []
        for col in self.df_orig.columns[2:]:
            assessment_list.extend(list(predictions[col]))
        truth_list = []
        for col in self.df_orig.columns[2:]:
            truth_list.extend(list(self.df_orig[col]))
        X = np.array(assessment_list)
        y = np.array(truth_list)
        X_ = (X[:,0] - X[:,1]).reshape(-1, 1)
        X = np.hstack((X, X_))
        X_ = (X[:,0] + X[:,1]).reshape(-1, 1)
        X = np.hstack((X, X_))
        X_ = (X[:,0] * X[:,1]).reshape(-1, 1)
        X_ = (X[:,0] * X[:,1] / (X[:,0] + X[:,1])).reshape(-1, 1)
        #X = np.hstack((X, X_))
        X_ = ((X[:,0] - X[:,1])/(X[:,0] + X[:,1])).reshape(-1, 1)
        #X = np.hstack((X, X_))
        #X = X[:,2:4]
        #y = y * y
        #X = X[y != 0]
        #y = y[y != 0]
        #X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        if create:
            clf = DecisionTreeClassifier(
                max_leaf_nodes=10000, 
                #random_state=0, 
                max_depth=5, 
                min_samples_leaf=10, 
                min_impurity_decrease=0.001
            )
            clf.fit(X, y)
            dump(clf, os.getcwd() + '/models/interpreter.clf') 
        else:
            clf = load(os.getcwd() + '/models/interpreter.clf') 
        max_ = np.argmax(clf.predict_proba(X, check_input=True), axis = 1).reshape(9,-1).T
        prob_preds = np.array([np.max(row) for row in clf.predict_proba(X, check_input=True)]).reshape(9,-1).T
        cols = list(preds.columns[1:])
        cols.extend([col + ' prob' for col in cols])
        output = pd.DataFrame(np.hstack((max_, prob_preds)), columns = cols)
        def num2ass(x):
            if x == 0:
                return 'Negative'
            elif x == 1:
                return ''
            else:
                return 'Positive'
        for col in preds.columns[1:]:
            output[col] = output[col].apply(lambda x: num2ass(x))
            output[col + ' prob'] = output[col + ' prob'].apply(lambda x: ' ' + str(int((2*x-1)*100)))
            output[col] = tuple(zip(output[col], output[col + ' prob']))
            output = output.drop([col + ' prob'], axis = 1)
        output['rev_content'] = list(self.df_orig['rev_content'])
        output = output[self.df_orig.columns[1:]]
        output.to_excel('output.xlsx')
        return output
        
    
    def output_generator(self, output_file_string  = 'output.xlsx'):
        learn_clf = self.predictor()
        tok_inf_df = tokenize_df(self.test_df, 'rev_content')
        dls_clf = self.model_data_prep(False, True)[0]
        inf_dl = learn_clf.dls.test_dl(tok_inf_df[0])
        all_predictions = learn_clf.get_preds(dl = inf_dl, reorder = False)
        probs = all_predictions[0].numpy()   
        indices = inf_dl.get_idxs()
        predictions = pd.DataFrame(all_predictions[0].numpy(), columns = learn_clf.dls.vocab[1])
        predictions['index'] = indices
        predictions = pd.merge(predictions, self.test_df[['index', 'rev_content']], on='index')[self.test_df.columns[:-1]]
        predictions = predictions.sort_values(by = ['index'])
        predictions = predictions.reset_index(drop = True)
        predictions = predictions.drop(['index'], axis = 1)
        #predictions.to_excel('test_predicted.xlsx')
        return self.output_interpreter(predictions)
    
    
    
               

