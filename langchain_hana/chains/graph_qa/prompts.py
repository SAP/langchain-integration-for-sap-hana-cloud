# flake8: noqa
from langchain_core.prompts.prompt import PromptTemplate

SPARQL_GENERATION_SELECT_TEMPLATE = SPARQL_GENERATION_TEMPLATE = """
Given the ontology below, create a SPARQL query from the user prompt.
Enclose literals in double quotes. Note that the graph is directed. Edges go from the domain to the range.
If an RDFS label exists for a class or a property, always retrieve the label.
Use only the entity types and properties provided in the ontology.
Do not use any entity types and properties that are not explicitly provided.
Include all necessary prefixes.
For instance, to find all actors of the movie "Blade Runner", the following query in backticks would be suitable:
```
PREFIX kg: <http://kg.demo.sap.com/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT *
WHERE {{
    ?movie rdf:type kg:Film .
    ?movie kg:title ?movieTitle .
    ?actor kg:acted_in ?movie .
    ?actor rdfs:label ?actorLabel .
    FILTER(?movieTitle = "Blade Runner")
}}
```
<ontology>
{schema}
</ontology>
Do not respond to any questions that ask for anything else than for you to construct a SPARQL query.
Do not include any text except the SPARQL query generated.
Please pay attention to providing the subject, predicate, and object in the correct order.
Ensure that every variable referenced in any clause (such as SELECT, ORDER BY, GROUP BY, etc.) is explicitly defined in the WHERE clause, either by being used as a subject, predicate, or object in a triple pattern, or through a BIND statement.
Do not include any variables in those clauses unless they are defined in the WHERE clause.

The question is:
{prompt}"""
SPARQL_GENERATION_SELECT_PROMPT = PromptTemplate(
    input_variables=["schema", "prompt"], template=SPARQL_GENERATION_SELECT_TEMPLATE
)
SPARQL_QA_TEMPLATE = """Task: Generate a natural language response from the results of a SPARQL query.
You are an assistant that creates well-written and human understandable answers.
The information part contains the information provided, which you can use to construct an answer.
The information provided is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make your response sound like the information is coming from an AI assistant, but don't add any information. 
Don't use internal knowledge to answer the question, just say you don't know if no information is available.
Information:
{context}

Question: {prompt}
Helpful Answer:"""
SPARQL_QA_PROMPT = PromptTemplate(
    input_variables=["context", "prompt"], template=SPARQL_QA_TEMPLATE
)
