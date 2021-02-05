from transformers import AutoTokenizer, AutoModelForTokenClassification

from transformers import pipeline

import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")

bert = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

datas = pd.read_excel('newData.xlsx')
newest_doc = datas['Use_Case_Description']

nlp = pipeline("ner", model=bert, tokenizer=tokenizer)
ind = 0
for index, row in datas.iterrows():
    ner_results = nlp(row['Use_Case_Description'])
    print(row['Use_Case_No'], ner_results)
    ind += 1



