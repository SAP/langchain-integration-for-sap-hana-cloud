"""Test fixtures for HanaSparqlQAAgent evaluations against the
``kgdocu_movies`` knowledge graph.

Four collections are provided:

* ``TRAJECTORY_EVALS`` / ``OFF_TOPIC_EVALS`` / ``FOLLOWUP_EVALS`` — each
  entry is ``(question, reference_trajectory, input_messages)``.
  ``input_messages`` is what to pass to
  ``agent.invoke({"messages": ...})``; ``reference_trajectory`` is the
  full ideal conversation (incl. tool calls) for trajectory evaluators.
* ``ANSWER_EVALS`` — ``(question, reference_answer)`` pairs for an
  LLM-as-judge over the agent's final natural-language answer.
"""

import json
import uuid
from textwrap import dedent
from typing import Any

REFERENCE_ONTOLOGY_CONTENT = """
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://kg.demo.sap.com/acted_in> a owl:ObjectProperty ;
    rdfs:label "acted_in" ;
    rdfs:domain <http://kg.demo.sap.com/Actor> ;
    rdfs:range <http://kg.demo.sap.com/Film> .

<http://kg.demo.sap.com/dateOfBirth> a owl:DatatypeProperty ;
    rdfs:label "dateOfBirth" ;
    rdfs:domain <http://kg.demo.sap.com/Actor> ;
    rdfs:range xsd:dateTime .

<http://kg.demo.sap.com/directed> a owl:ObjectProperty ;
    rdfs:label "directed" ;
    rdfs:domain <http://kg.demo.sap.com/Director> ;
    rdfs:range <http://kg.demo.sap.com/Film> .

<http://kg.demo.sap.com/genre> a owl:ObjectProperty ;
    rdfs:label "genre" ;
    rdfs:domain <http://kg.demo.sap.com/Film> ;
    rdfs:range <http://kg.demo.sap.com/Genre> .

<http://kg.demo.sap.com/placeOfBirth> a owl:ObjectProperty ;
    rdfs:label "placeOfBirth" ;
    rdfs:domain <http://kg.demo.sap.com/Actor> ;
    rdfs:range <http://kg.demo.sap.com/Place> .

<http://kg.demo.sap.com/title> a owl:DatatypeProperty ;
    rdfs:label "title" ;
    rdfs:domain <http://kg.demo.sap.com/Film> ;
    rdfs:range xsd:string .

rdfs:label a owl:DatatypeProperty ;
    rdfs:label "label" ;
    rdfs:domain <http://kg.demo.sap.com/Actor>,
        <http://kg.demo.sap.com/Director>,
        <http://kg.demo.sap.com/Genre>,
        <http://kg.demo.sap.com/Place> ;
    rdfs:range xsd:string .

<http://kg.demo.sap.com/Director> a owl:Class ;
    rdfs:label "Director" .

<http://kg.demo.sap.com/Genre> a owl:Class ;
    rdfs:label "Genre" .

<http://kg.demo.sap.com/Place> a owl:Class ;
    rdfs:label "Place" .

<http://kg.demo.sap.com/Actor> a owl:Class ;
    rdfs:label "Actor" .

<http://kg.demo.sap.com/Film> a owl:Class ;
    rdfs:label "Film" .
"""


def _tool_call_id() -> str:
    """Return a fresh OpenAI-style ``call_…`` tool-call id."""
    return f"call_{uuid.uuid4().hex[:24]}"


def _ontology_call() -> list[dict[str, Any]]:
    call_id = _tool_call_id()
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "retrieve_ontology",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "retrieve_ontology",
            "tool_call_id": call_id,
            "content": REFERENCE_ONTOLOGY_CONTENT,
        },
    ]


def _sparql_call(query: str, result: str) -> list[dict[str, Any]]:
    call_id = _tool_call_id()
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "execute_sparql",
                        "arguments": json.dumps({"query": query}),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "execute_sparql",
            "tool_call_id": call_id,
            "content": result,
        },
    ]


def _message(role: str, content: str) -> dict[str, Any]:
    """Build a plain text message dict ``{"role": role, "content": content}``."""
    return {"role": role, "content": content}


# Single-turn questions: agent should call retrieve_ontology, then
# execute_sparql, then answer.
TRAJECTORY_EVALS: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = [
    # 1. Simple "actors of a movie"
    (
        "Which actors acted in Blade Runner?",
        [
            _message("user", "Which actors acted in Blade Runner?"),
            *_ontology_call(),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?actorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie rdf:type kg:Film .
                  ?movie kg:title "Blade Runner" .
                  ?actor kg:acted_in ?movie .
                  ?actor rdfs:label ?actorLabel .
              }
            """),
                result=dedent("""
                actorLabel
                Harrison Ford
                Rutger Hauer
                Joe Turkel
                William Sanderson
                Morgan Paull
                Hy Pyke
                Brion James
                Sean Young
                Edward James Olmos
                James Hong
                Joanna Cassidy
                Daryl Hannah
                M. Emmet Walsh
            """),
            ),
            _message(
                "assistant",
                (
                    "The actors in Blade Runner include Harrison Ford "
                    "(Rick Deckard), Rutger Hauer (Roy Batty), Sean Young, "
                    "Edward James Olmos (Gaff), Daryl Hannah, Joanna "
                    "Cassidy (Zhora Salome), Brion James (Leon Kowalski), "
                    "William Sanderson (J. F. Sebastian), Joe Turkel (Eldon "
                    "Tyrell), James Hong (Hannibal Chew), M. Emmet Walsh "
                    "(Harry Bryant), Morgan Paull (Dave Holden) and Hy Pyke "
                    "(Taffey Lewis)."
                ),
            ),
        ],
        [_message("user", "Which actors acted in Blade Runner?")],
    ),
    # 2. Director of a movie
    (
        "Who directed Blade Runner?",
        [
            _message("user", "Who directed Blade Runner?"),
            *_ontology_call(),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?directorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie kg:title "Blade Runner" .
                  ?director kg:directed ?movie .
                  ?director rdfs:label ?directorLabel .
              }
            """),
                result=dedent("""
                directorLabel
                Ridley Scott
            """),
            ),
            _message("assistant", "Blade Runner was directed by Ridley Scott."),
        ],
        [_message("user", "Who directed Blade Runner?")],
    ),
    # 3. Graph-shaped question whose answer is NOT in the graph.
    # The agent should still call ontology + sparql, get an empty result,
    # and report that the film isn't in the graph.
    (
        "Who directed Inception?",
        [
            _message("user", "Who directed Inception?"),
            *_ontology_call(),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?directorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie kg:title "Inception" .
                  ?director kg:directed ?movie .
                  ?director rdfs:label ?directorLabel .
              }
            """),
                result=dedent("""
                directorLabel
            """),
            ),
            _message(
                "assistant",
                (
                    "I couldn't find Inception in the movies knowledge graph, "
                    "so I don't have a director for it."
                ),
            ),
        ],
        [_message("user", "Who directed Inception?")],
    ),
    # 4. Cross-movie aggregation — actors that appear in BOTH Matrix films
    (
        "Which actors appeared in both The Matrix and The Matrix Reloaded?",
        [
            _message(
                "user",
                (
                    "Which actors appeared in both The Matrix and "
                    "The Matrix Reloaded?"
                ),
            ),
            *_ontology_call(),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?actorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?m1 kg:title "The Matrix" .
                  ?m2 kg:title "The Matrix Reloaded" .
                  ?actor kg:acted_in ?m1 .
                  ?actor kg:acted_in ?m2 .
                  ?actor rdfs:label ?actorLabel .
              }
            """),
                result=dedent("""
                actorLabel
                Keanu Reeves
                Laurence Fishburne
                Carrie-Anne Moss
                Hugo Weaving
                Gloria Foster
            """),
            ),
            _message(
                "assistant",
                (
                    "Actors appearing in both The Matrix and The Matrix "
                    "Reloaded include Keanu Reeves, Laurence Fishburne, "
                    "Carrie-Anne Moss, Hugo Weaving and Gloria Foster."
                ),
            ),
        ],
        [
            _message(
                "user",
                "Which actors appeared in both The Matrix and The Matrix Reloaded?",
            )
        ],
    ),
]


# Off-topic questions: agent should answer directly without calling any tools.
OFF_TOPIC_EVALS: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = [
    # 1. Pure greeting / small talk
    (
        "Hello! How are you today?",
        [
            _message("user", "Hello! How are you today?"),
            _message("assistant", "Hello! How can I help you today?"),
        ],
        [_message("user", "Hello! How are you today?")],
    ),
    # 2. General world knowledge that is NOT in the graph
    (
        "What is the capital of France?",
        [
            _message("user", "What is the capital of France?"),
            _message(
                "assistant",
                (
                    "I don't know — that information isn't available in the "
                    "movies knowledge graph I have access to."
                ),
            ),
        ],
        [_message("user", "What is the capital of France?")],
    ),
    # 3. Math — no need to query the KG
    (
        "What is 2 + 2?",
        [
            _message("user", "What is 2 + 2?"),
            _message(
                "assistant", "I don't have the knowledge to answer this question."
            ),
        ],
        [_message("user", "What is 2 + 2?")],
    ),
    # 4. Meta question about the agent itself
    (
        "What can you help me with?",
        [
            _message("user", "What can you help me with?"),
            _message(
                "assistant",
                (
                    "I can answer questions about a movies knowledge graph "
                    "covering films, actors, directors, roles, genres and "
                    "places of birth."
                ),
            ),
        ],
        [_message("user", "What can you help me with?")],
    ),
]


# Multi-turn: turn-1 fetches ontology + sparql; turn-2 should reuse the
# ontology and only call execute_sparql.
FOLLOWUP_EVALS: list[tuple[str, list[dict[str, Any]], list[dict[str, Any]]]] = [
    # 1. First "who acted in Blade Runner?" then follow-up "and The Matrix?"
    (
        "actors_blade_runner_then_matrix",
        [
            # --- Turn 1 -------------------------------------------------------
            _message("user", "Which actors acted in Blade Runner?"),
            *_ontology_call(),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?actorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie kg:title "Blade Runner" .
                  ?actor kg:acted_in ?movie .
                  ?actor rdfs:label ?actorLabel .
              }
            """),
                result=dedent("""
                actorLabel
                Harrison Ford
                Rutger Hauer
                Sean Young
                Edward James Olmos
            """),
            ),
            _message(
                "assistant",
                (
                    "Actors in Blade Runner include Harrison Ford, Rutger "
                    "Hauer, Sean Young and Edward James Olmos, among others."
                ),
            ),
            # --- Turn 2 (follow-up) -------------------------------------------
            # NOTE: no _ontology_call() here — ontology is already in context.
            _message("user", "And in The Matrix?"),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?actorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie kg:title "The Matrix" .
                  ?actor kg:acted_in ?movie .
                  ?actor rdfs:label ?actorLabel .
              }
            """),
                result=dedent("""
                actorLabel
                Keanu Reeves
                Laurence Fishburne
                Carrie-Anne Moss
                Hugo Weaving
            """),
            ),
            _message(
                "assistant",
                (
                    "Actors in The Matrix include Keanu Reeves, Laurence "
                    "Fishburne, Carrie-Anne Moss and Hugo Weaving, among "
                    "others."
                ),
            ),
        ],
        # input_messages: full turn-1 trajectory + turn-2 user message.
        [
            _message("user", "Which actors acted in Blade Runner?"),
            *_ontology_call(),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?actorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie kg:title "Blade Runner" .
                  ?actor kg:acted_in ?movie .
                  ?actor rdfs:label ?actorLabel .
              }
            """),
                result=dedent("""
                actorLabel
                Harrison Ford
                Rutger Hauer
                Sean Young
                Edward James Olmos
            """),
            ),
            _message(
                "assistant",
                (
                    "Actors in Blade Runner include Harrison Ford, Rutger "
                    "Hauer, Sean Young and Edward James Olmos, among others."
                ),
            ),
            _message("user", "And in The Matrix?"),
        ],
    ),
    # 2. First "who directed Blade Runner?" then "and The Matrix?"
    (
        "director_blade_runner_then_matrix",
        [
            # --- Turn 1 -------------------------------------------------------
            _message("user", "Who directed Blade Runner?"),
            *_ontology_call(),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?directorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie kg:title "Blade Runner" .
                  ?director kg:directed ?movie .
                  ?director rdfs:label ?directorLabel .
              }
            """),
                result=dedent("""
                directorLabel
                Ridley Scott
            """),
            ),
            _message("assistant", "Blade Runner was directed by Ridley Scott."),
            # --- Turn 2 (follow-up) -------------------------------------------
            _message("user", "And who directed The Matrix?"),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?directorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie kg:title "The Matrix" .
                  ?director kg:directed ?movie .
                  ?director rdfs:label ?directorLabel .
              }
            """),
                result=dedent("""
                directorLabel
                Lilly Wachowski
                Lana Wachowski
            """),
            ),
            _message(
                "assistant",
                ("The Matrix was directed by Lilly Wachowski and " "Lana Wachowski."),
            ),
        ],
        # input_messages: full turn-1 trajectory + turn-2 user message.
        [
            _message("user", "Who directed Blade Runner?"),
            *_ontology_call(),
            *_sparql_call(
                query=dedent("""
              PREFIX kg: <http://kg.demo.sap.com/>
              PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
              SELECT ?directorLabel
              FROM <kgdocu_movies>
              WHERE {
                  ?movie kg:title "Blade Runner" .
                  ?director kg:directed ?movie .
                  ?director rdfs:label ?directorLabel .
              }
            """),
                result=dedent("""
                directorLabel
                Ridley Scott
            """),
            ),
            _message("assistant", "Blade Runner was directed by Ridley Scott."),
            _message("user", "And who directed The Matrix?"),
        ],
    ),
]


# Content-level evaluation of the agent's final answer (e.g. with an LLM-as-judge).
ANSWER_EVALS: list[tuple[str, str]] = [
    (
        "Who directed The Matrix?",
        "The Matrix was directed by Lilly Wachowski and Lana Wachowski.",
    ),
    (
        "Who directed Blade Runner?",
        "Blade Runner was directed by Ridley Scott.",
    ),
    (
        "Who directed The Matrix Reloaded?",
        "The Matrix Reloaded was directed by Lilly Wachowski and Lana Wachowski.",
    ),
    (
        "Where was Keanu Reeves born?",
        "Keanu Reeves was born in Beirut.",
    ),
    (
        "Where was Hugo Weaving born?",
        "Hugo Weaving was born in Ibadan.",
    ),
    (
        "When was Keanu Reeves born?",
        "Keanu Reeves was born on 2 September 1964.",
    ),
    (
        "How many films are in the knowledge graph?",
        "There are 3 films in the knowledge graph.",
    ),
    (
        "List all the films in the knowledge graph.",
        (
            "The films in the knowledge graph are: The Matrix, "
            "The Matrix Reloaded, and Blade Runner."
        ),
    ),
    (
        "Which films belong to the cyberpunk genre?",
        (
            "The Matrix, The Matrix Reloaded, and Blade Runner are all "
            "classified as cyberpunk films."
        ),
    ),
    (
        "What genres does Blade Runner belong to?",
        (
            "Blade Runner's genres include science fiction, cyberpunk, "
            "dystopian, thriller, action, tech noir, neo-noir, drama, "
            "film noir, arthouse science fiction, and film based on a "
            "novel."
        ),
    ),
    (
        "Which films did Hugo Weaving act in?",
        "Hugo Weaving acted in The Matrix and The Matrix Reloaded.",
    ),
    (
        "Which actors appeared in both The Matrix and The Matrix Reloaded?",
        (
            "Actors in both films include Keanu Reeves, Laurence "
            "Fishburne, Carrie-Anne Moss, Hugo Weaving and Gloria Foster."
        ),
    ),
    (
        "Who is the oldest actor in The Matrix?",
        "Steve Dodd, born on 1 June 1928, is the oldest actor in The Matrix.",
    ),
    (
        "Who is the youngest actor in The Matrix?",
        "Rowan Witt, born on 5 November 1988, is the youngest actor in The Matrix.",
    ),
    (
        "Is Blade Runner a science fiction film?",
        "Yes, Blade Runner is classified as a science fiction film.",
    ),
]
