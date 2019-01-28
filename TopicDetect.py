import matplotlib.pyplot as plt
import os, re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from nltk.corpus import stopwords
from string import punctuation,digits
from nltk.tokenize import word_tokenize
from datetime import datetime as dt
from nltk.probability import FreqDist
import nltk.data


train_arr = []
test_arr = []
train_lbl = []
test_lbl = []
i = 0
files_train = {}
files_test = {}
root_train = "\\Users\\vikash\\Desktop\\Python data science\\20news-bydate\\20news-bydate-train\\"
class_titles_train = os.listdir(root_train)
root_test = "\\Users\\vikash\\Desktop\\Python data science\\20news-bydate\\20news-bydate-test\\"
class_titles_test = os.listdir(root_test)
metrics_dict = []
example3 = 'C:\\Users\\vikash\\Desktop\\Python data science\\test2.txt'
example2 = 'C:\\Users\\vikash\\Desktop\\Python data science\\test.txt'
example = '\\Users\\vikash\\Desktop\\Python data science\\20news-bydate\\20news-bydate-test\\sci.electronics\\53984'
pattern = re.compile(r'([a-zA-Z]+|[0-9]+(\.[0-9]+)?)')
def main():
     get_train_data()
     get_test_data()
     print("Creating arrays...")
     creating_arrays()
     print("Testing BNB:\n")
     create_text_vect_tfidf_BNB()
     print("Testing GNB:\n")
     create_text_vect_tfidf_GNB()
     print("Testing MNB:\n")
     create_text_vect_tfidf_MNB()
     print("Testing KNN:\n")
     knn()
     print("Topic Detect Example 2:\n")
     Topic_detect_MnB(example2)    
     f = open(example2)
     y1 = f.read() 
     print("Summary of Example 2:\n")
     Summ(y1,3)
     print("Actual Example 2:\n")
     print(y1)
     f.close
     g = open(example3)
     Topic_detect_MnB(example3)
     y2 = g.read() 
     print("Summary of Example 3:\n")
     Summ(y2,5)
     print("Actual Example 3:\n")
     print(y2)
     g.close
    
   # tfidf_trans()
#    print(test_arr)
#    print(clean_text(example))
#    f = open(example)
#    print(f.read())


def get_train_data():
    root_train = "\\Users\\vikash\\Desktop\\Python data science\\20news-bydate\\20news-bydate-train\\"
    ##print(root_train)
    folders = [root_train + folder + '\\' for folder in os.listdir(root_train)]
    ##print(folders[0])
    class_titles = os.listdir(root_train)
    ##print(class_titles)
    
    for folder, title in zip(folders, class_titles):
        files_train[title] = [folder + f for f in os.listdir(folder)]
    ##print (files)


def get_test_data():
    root_test = "\\Users\\vikash\\Desktop\\Python data science\\20news-bydate\\20news-bydate-test\\"
    ##print(root_test)
    folders_test  = [root_test + folder + '\\' for folder in os.listdir(root_test)]
    ##print(folders_test)    
    class_titles = os.listdir(root_test)
    for folder, title in zip(folders_test, class_titles):
        files_test[title] = [folder + f for f in os.listdir(folder)]
    ##print(files_test)


def clean_text(path):
    text_translated = ''
    try:
        f = open(path)
        raw = f.read().lower()
        text = pattern.sub(r' \1 ', raw.replace('\n', ' '))
        text_translated = text.translate(str.maketrans('','',punctuation + digits))
        text_translated = ' '.join([word for word in text_translated.split(' ') if (word and len(word) > 1)])
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text_translated)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        filtered_sentence = []
        for w in word_tokens:
            if w not in stop_words:
                filtered_sentence.append(w)
        final_op = ' '.join(filtered_sentence)
    
        
    finally:
        f.close()
    return final_op


def creating_arrays():
    
    for i in range(10):
        for path in files_train[class_titles_train[i]]:
            #  print(path)
            #print(class_titles_train[i])
            train_arr.append(clean_text(path))
            train_lbl.append(class_titles_train[i])
    for i in range(10):
        for path in files_test[class_titles_test[i]]:
            #  print(path)
            #print(class_titles_train[i])
            test_arr.append(clean_text(path))
            test_lbl.append(class_titles_test[i])


def create_text_vect_tfidf_BNB():
    vectorizer = CountVectorizer()
    vectorizer.fit(train_arr)
    train_mat = vectorizer.transform(train_arr)
  #  print(train_mat.shape)
    test_mat = vectorizer.transform(test_arr)
 #   print(test_mat.shape)
    tfidf = TfidfTransformer()
    tfidf.fit(train_mat)
    train_tfmat = tfidf.transform(train_mat)
    print(train_tfmat.shape)
    test_tfmat = tfidf.transform(test_mat)
    print(test_tfmat.shape)
    bnb = BernoulliNB()
    bnb_me = Metrics_Classifier(train_tfmat, train_lbl, test_tfmat, test_lbl, bnb)
    metrics_dict.append({'name':'BernoulliNB', 'metrics':bnb_me})
    
def Metrics_Classifier(x_train, y_train, x_test, y_test, clf):
    metrics = []
    start = dt.now()
    clf.fit(x_train, y_train)
    end = dt.now()
    print ('training time: ', (end - start))
    
    # add training time to metrics
    metrics.append(end-start)
    
    start = dt.now()
    yhat = clf.predict(x_test)
    print('teh predict i guess')
    print(yhat)
    end = dt.now()
    print ('testing time: ', (end - start))
    
    # add testing time to metrics
    metrics.append(end-start)
    
    print ('classification report: \n')
#     print classification_report(y_test, yhat)
    print(classification_report(y_test, yhat))
    
    print ('f1 score')
    print (f1_score(y_test, yhat, average='macro'))
    
    print ('accuracy score: \n')
    print (accuracy_score(y_test, yhat))
    
    precision = precision_score(y_test, yhat, average=None)
    recall = recall_score(y_test, yhat, average=None)
    
    # add precision and recall values to metrics
    for p, r in zip(precision, recall):
        metrics.append(p)
        metrics.append(r)
    
    
    #add macro-averaged F1 score to metrics
    metrics.append(f1_score(y_test, yhat, average='macro'))
    
    print ('confusion matrix:')
    print (confusion_matrix(y_test, yhat))
    
    # plotting the confusion matrix
    plt.imshow(confusion_matrix(y_test, yhat), interpolation='nearest')
    plt.show()
    
    return metrics

def create_text_vect_tfidf_GNB():
    vectorizer = CountVectorizer()
    vectorizer.fit(train_arr)
    train_mat = vectorizer.transform(train_arr)
  #  print(train_mat.shape)
    test_mat = vectorizer.transform(test_arr)
 #   print(test_mat.shape)
    tfidf = TfidfTransformer()
    tfidf.fit(train_mat)
    train_tfmat = tfidf.transform(train_mat)
    print(train_tfmat.shape)
    test_tfmat = tfidf.transform(test_mat)
    print(test_tfmat.shape)
    gnb = GaussianNB()
    gnb_me = Metrics_Classifier(train_tfmat.toarray(), train_lbl, test_tfmat.toarray(), test_lbl, gnb)
    metrics_dict.append({'name':'GaussianNB', 'metrics':gnb_me})
    
def create_text_vect_tfidf_MNB():
    vectorizer = CountVectorizer()
    vectorizer.fit(train_arr)
    train_mat = vectorizer.transform(train_arr)
 #  print(train_mat.shape)
    test_mat = vectorizer.transform(test_arr)
 #   print(test_mat.shape)
    tfidf = TfidfTransformer()
    tfidf.fit(train_mat)
    train_tfmat = tfidf.transform(train_mat)
    print(train_tfmat.shape)
    test_tfmat = tfidf.transform(test_mat)
    print(test_tfmat.shape)
    mnb = MultinomialNB()
    mnb_me = Metrics_Classifier(train_tfmat.toarray(), train_lbl, test_tfmat.toarray(), test_lbl, mnb)
    metrics_dict.append({'name':'MultinomialNB', 'metrics':mnb_me})
def knn():
    vectorizer = CountVectorizer()
    vectorizer.fit(train_arr)
    train_mat = vectorizer.transform(train_arr)
  #  print(train_mat.shape)
    test_mat = vectorizer.transform(test_arr)
 #   print(test_mat.shape)
    tfidf = TfidfTransformer()
    tfidf.fit(train_mat)
    train_tfmat = tfidf.transform(train_mat)
    print(train_tfmat.shape)
    test_tfmat = tfidf.transform(test_mat)
    print(test_tfmat.shape)
    for nn in [10]:
        print ('knn with ', nn, ' neighbors')
        knn = KNeighborsClassifier(n_neighbors=nn)
        knn_me = Metrics_Classifier(train_tfmat, train_lbl, test_tfmat, test_lbl, knn)
        metrics_dict.append({'name':'5NN', 'metrics':knn_me})
        print (' ')
   
def Topic_detect_MnB(path):
    vectorizer = CountVectorizer()
    vectorizer.fit(train_arr)
    train_mat = vectorizer.transform(train_arr)
 #  print(train_mat.shape)
    new_arr = []
    basic_arr = []
    f = open(path)
    basic_arr.append(f.read())
    
    new_arr.append(clean_text(path))
    test_mat = vectorizer.transform(new_arr)
 #   print(test_mat.shape)
    tfidf = TfidfTransformer()
    tfidf.fit(train_mat)
    train_tfmat = tfidf.transform(train_mat)
    print(train_tfmat.shape)
    test_tfmat = tfidf.transform(test_mat)
    print(test_tfmat)
    print(test_tfmat.shape)
    mnb = MultinomialNB()
#    Main_Classifier(train_tfmat.toarray(), train_lbl, test_tfmat.toarray(),mnb)
    mnb.fit(train_tfmat.toarray(),train_lbl)
    yhat = mnb.predict(test_tfmat.toarray())
    print("Output Predict:")
    print(yhat)
    
def Summ(arr,num_sent):
    raw = arr.lower()
    text = pattern.sub(r' \1 ', raw.replace('\n', ' '))
    text_translated = text.translate(str.maketrans('','',punctuation + digits))
    text_translated = ' '.join([word for word in text_translated.split(' ') if (word and len(word) > 1)])
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text_translated)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    #print(filtered_sentence)
    word_frequencies = FreqDist(filtered_sentence)
    most_freq_words =[]
   # print(word_frequencies.items())
    s = [(k, word_frequencies[k]) for k in sorted(word_frequencies, key=word_frequencies.get, reverse=True)]
    for k, v in s:
        most_freq_words.append(k)
    #print(most_freq_words[:50])
    use =  most_freq_words[:50]
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    actual_sentences = sent_detector.tokenize(arr)
    working_sentences = [sentence.lower()
        for sentence in actual_sentences]
   # print(actual_sentences)
   # print("trololol")
   # print(working_sentences)
    output_sentences = []
    for word in use:
        for i in range(0, len(working_sentences)):
            if (word in working_sentences[i]
                and actual_sentences[i] not in output_sentences):
                    output_sentences.append(actual_sentences[i])
                    break
            if len(output_sentences) >= num_sent: break
        if len(output_sentences) >= num_sent: break
    
    #print(output_sentences)
    final_op =[]
    for sentence in actual_sentences:
        for sent2 in output_sentences:
            if(sentence==sent2):
                final_op.append(sent2)
    print( " ".join(final_op))
                
  #  for k, v in od.items(): print(k, v)
  
if __name__ == '__main__':
    main()