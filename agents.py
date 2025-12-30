from langchain.tools import tool
import requests
from RAG import load_dictionary_vectorstore
from langchain.agents import create_agent
from langchain_ollama import ChatOllama



@tool
def get_tags(text: str) -> str:
    """get tags for the old french phrase using Udpipe API."""
    API_URL = "https://lindat.mff.cuni.cz/services/udpipe/api/process"
    data = {
        "data": text,
        "model": "old_french-profiterole-ud-2.17-251125",
        "tokenizer":"",
        "tagger":"",
        "parser":""
    }
    response = requests.post(API_URL, data=data)
    response.raise_for_status()
    output = response.json()["result"]
    return output



def make_call_dictionary_agent(dictionary_agent):
    @tool
    def call_dictionary_agent(text: str) -> str:
        """Lookup dictionary to get definitions for text."""
        response = dictionary_agent.invoke(
            {"messages": [{"role": "user", "content": text}]}
        )
        return response["messages"][-1].content

    return call_dictionary_agent

def make_retrieve_dict_tool(retriever):
    @tool(response_format="content_and_artifact")
    def retrieve_dict(word: str): 
        """Retrieve Dictionary explanations for a word."""
        docs = retriever.invoke(word)
        serialized = "\n\n".join(doc.page_content for doc in docs)
        return serialized, docs

    return retrieve_dict

def translation_pipeline(old_french_text, retriever):
    
    llama = ChatOllama(model="llama3.1:8b", temperature=0.1)

    retrieve_dict = make_retrieve_dict_tool(retriever)
    # dictionary agent
    dictionary_agent = create_agent(
        model=llama,
        name="Dictionary Agent",
        tools=[retrieve_dict],
        system_prompt="""
        You are a historical linguistics assistant specialized in Old French.

Given an Old French phrase as input:
1. Identify important or ambiguous words (MAX 5) in the input (verbs, nouns, rare forms, idioms).
2. For EACH identified word:
   a. Call retrieve_dict(word) to look up dictionary definitions.
   b. If multiple lemmas or meanings exist, include all of them.
3. Repeat tool calls until ALL identified words have been looked up.


Output ONLY a list of short definitions and explanations.
Do not add commentary, introductions, or conclusions.

        """,
        )

    #  Create a tool bound to dictionary_agent
    call_dictionary_agent = make_call_dictionary_agent(dictionary_agent)

    #  Translation agent SECOND
    translation_agent = create_agent(
        model=llama,
        tools=[get_tags, call_dictionary_agent],
        system_prompt="""
    You are a historical linguistics expert.
    Translate the following Old French input into Modern English.
    Preserve meaning and avoid modernization errors.
    you have access to 'get_tags(input)' tool that provides POS tags and syntactic dependencies for Old French text.
    you also have access to call_dictionary_agent(text) which takes the old french phrase as input and provides dictionary definitions.
    output the translation ONLY. NO additional commentary.
    """,
    )
    

    # Invoke
    return translation_agent.invoke(
        {"messages": [{"role": "user", "content": old_french_text}]}
    )


    






if __name__ == "__main__":
    old_french_text = "Lanselos, ki de nule riens ne les doute, ains les bee tous a metre a desconfiture, s’il onques puet, lors laisse courre tout maintenant."
    vectordb = load_dictionary_vectorstore("./chroma_atilf_en_dict")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    translation_response = translation_pipeline(old_french_text, retriever)
    for res in translation_response['messages']:
        print(res.content)
    
    
    