from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
# from langchain.output_parsers import PydanticOutputParser


import getpass
import os
from dotenv import load_dotenv
load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    API_KEY = getpass.getpass("Enter your Google API Key: ")
    os.environ["GOOGLE_API_KEY"] = API_KEY
API_KEY = os.getenv("GOOGLE_API_KEY")

eval_schema = {
    "type": "object",
    "description": "Binary error annotation for Old French → Modern English translation evaluation.",
    "properties": {
        "role_coref": {
            "type": "integer",
            "description": "Argument structure or discourse coreference error (0=no, 1=yes)"
        },
        "verb": {
            "type": "integer",
            "description": "Verb tense/aspect/mood error (0=no, 1=yes)"
        },
        "lexical_idiom": {
            "type": "integer",
            "description": "Lexical sense or idiomatic mistranslation (0=no, 1=yes)"
        },
        "modern": {
            "type": "integer",
            "description": "Inappropriate modernization or archaization (0=no, 1=yes)"
        },
        "content": {
            "type": "integer",
            "description": "Content distortion, omission, or addition (0=no, 1=yes)"
        },
        "hallucination": {
            "type": "integer",
            "description": "Unsupported or invented content (0=no, 1=yes)"
        },
        "neg": {
            "type": "integer",
            "description": "Negation misinterpretation (0=no, 1=yes)"
        },
        "literacy": {
            "type": "integer",
            "description": "Fluency, coherence, or stylistic inconsistency (0=no, 1=yes)"
        },
        "overall_score": {
            "type": "integer",
            "description": "Overall translation quality score (1–10)"
        },
        "explanation": {
            "type": "string",
            "description": "Brief justification of the overall evaluation"
        }
    },
    "required": [
        "role_coref",
        "verb",
        "lexical_idiom",
        "modern",
        "content",
        "hallucination",
        "neg",
        "literacy",
        "overall_score",
        "explanation"
    ]
}



def eval_pipeline(original, gold_translation, agent_translation):
    prompt = """
You are a historical linguistics expert evaluating Old French → Modern English translations.

You are given:
(1) the original Old French passage,
(2) a gold reference translation,
(3) a system-generated translation.

Your task is to annotate whether specific translation error types are present in the system output.
Error categories are NOT mutually exclusive: multiple errors may co-occur in the same passage.

Mark each metric as:
0 = no error detected
1 = error present

Error categories:

1. role_coref (Role assignment + coreference):
   Errors that change “who did what to whom,” including subject–object reversals,
   incorrect argument–predicate mapping, or incorrect resolution of pronouns,
   possessives, or definite descriptions across clauses or discourse.

2. verb (Verb tense/aspect/mood):
   Mistranslation of tense, aspect, or mood that alters event structure,
   temporal relations, modality, or factuality.

3. lexical_idiom (Lexical sense / idiom):
   Incorrect lexical sense selection, failure to capture historical polysemy,
   or non-idiomatic rendering that distorts meaning.

4. modern (Modernization):
   Inappropriate modernization (overly contemporary phrasing) or excessive
   archaization that misrepresents the register or meaning.

5. content (Content fidelity):
   Omission, addition, or distortion of propositional content relative to the source.

6. hallucination:
   Introduction of information, entities, events, or relations not supported
   by the Old French source.

7. neg (Negation):
   Errors involving negation, including loss, addition, or scope misinterpretation.

8. literacy (Literacy & style):
   Problems with fluency, coherence, or stylistic consistency in Modern English,
   independent of semantic accuracy.

Finally, assign an overall quality score from 1 (very poor) to 10 (near-perfect),
and briefly justify the score with reference to the most salient errors.
Return ONLY the JSON object specified by the schema.
"""




    
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=API_KEY, 
    vertexai=False, 
)
    # create agent with structured output
    eval_agent = create_agent(
        model=llm,
        response_format=eval_schema,
        system_prompt=prompt,
    )

    result = eval_agent.invoke({"messages": [{"role": "user", "content": f"gold_translation: {gold_translation} agent_translation: {agent_translation}"}]})
    return result["structured_response"]

        
if __name__ == "__main__":
    df = pd.read_csv("eval_with_llm_eval.csv")

    # for index, row in df[:5].iterrows():
    #     original = row['original']
    #     gold_translation = row['gold_trans']
    #     agent_translation = row['translation']
    #     eval_result = eval_pipeline(original, gold_translation, agent_translation)
        
    #     for key, value in eval_result.items():
    #         df.at[index, f'llm_{key}'] = value
    #     df.to_csv("eval_with_llm_eval.csv", index=False)
    #     print(f"Processed row {index+1}, score: {eval_result['overall_score']}")  
    
    
   
    original="Et quant Lanselos l’a abatu, il nel regarde plus, non plus que s’il ne l’eüst onques veü, mais la u il voit les autres cevaliers, ki ja estoient monté et estoient issu des paveillons tout apareillié de ferir et de grever Lanselot se il peüssent. Lanselos, ki de nule riens ne les doute, ains les bee tous a metre a desconfiture, s’il onques puet, lors laisse courre tout maintenant."
    gold_translation = "And when Lancelot had felled him he no longer looked at him, as if he had not even seen him. But he sees the other knights already in the saddle, outside the pavilions, ready to strike and wound him if they can. Lancelot, who does not fear them at all, longs to defeat them if possible. He then charges off at full speed."
    agent_translation = "And when Lancelot had beaten him, he no longer looked at him, no more than if he had never seen him, but at the other knights, who were already mounted and had come out of the pavilions all armed to strike and harm Lancelot if they could. Lancelot, who doubted nothing at all, but rather saw them all as being destined to be defeated, if he could, now lets them run."
    eval_result = eval_pipeline(original, gold_translation, agent_translation)
    print(eval_result)