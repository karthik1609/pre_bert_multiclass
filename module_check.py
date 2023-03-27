import preprocess

phraser = preprocess.preprocess('Fat cat is good🙂 but bad car is red.')

sentiment_list = []
for phrase in phraser.phrase_extract():
    sentiment_list.append((phrase, preprocess.preprocess(phrase).sent_finder()))
    
print(phraser.phrase_extract())    
print(sentiment_list)
