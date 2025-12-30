import pandas as pd
from agents import translation_pipeline
from RAG import load_dictionary_vectorstore

# Load vector store and retriever
vectordb = load_dictionary_vectorstore("./chroma_atilf_en_dict")
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
#load phrases
df = pd.read_csv("eval.csv")


for index, row in df[:1].iterrows():
    old_french_text = row['original']
    translation_response = translation_pipeline(old_french_text, retriever)
    translation = translation_response['messages'][-1].content
    df.at[index, 'translation'] = translation
    print(old_french_text)
    print(translation)
    # print(f"Processed row {index+1}, size of original: {len(old_french_text)}, size of translation: {len(translation)}")
    # df.to_csv("eval_with_translations.csv", index=False)