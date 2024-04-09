import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

path = '/kaggle/input/log-files/log_files'                     

labels = []
text = []

for filename in os.listdir(path):                              
    if "Good" in filename:                                     
        labels.append("1")                                     
    else:                                                      
        labels.append("-1")                                    
    filename = os.path.join(path, filename)                    
    print (filename)                                           
    with open(filename, encoding = 'utf-8') as f:              
        content = f.read()                                     
    content.replace(",", " ")                                  
    content.replace('"', " ")
    text.append(content)                                       
    
vectorizer = CountVectorizer(stop_words = 'english', max_features=1000)

X_train_vectorized = vectorizer.fit_transform(text)

feature_names = vectorizer.get_feature_names_out()
print(feature_names)                              

features = pd.DataFrame(X_train_vectorized.toarray())      
features.columns = feature_names                           
features.index = labels                                    
print(features)                                            
features.to_csv('/kaggle/working/CSC 428 - Tim Leonhardt[Lab 5 - features].csv', index = True) 