"""Agent evaluations for HanaSparqlQAAgent using agentevals."""

import json
import os
from typing import Annotated, Any, TypedDict

import pytest
from agentevals.trajectory.llm import (  # type: ignore[import-untyped]
    TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
    create_trajectory_llm_as_judge,
)
from agentevals.trajectory.match import (  # type: ignore[import-untyped]
    create_trajectory_match_evaluator,
)
from gen_ai_hub.proxy.langchain import init_llm  # type: ignore[import-untyped]
from hdbcli import dbapi
from langchain_core.language_models import BaseChatModel

from langchain_hana import HanaRdfGraph, HanaSparqlQAAgent
from tests.fixtures.agent_evals_fixtures import (
    ANSWER_EVALS,
    FOLLOWUP_EVALS,
    OFF_TOPIC_EVALS,
    TRAJECTORY_EVALS,
)


class Config:
    def __init__(self) -> None:
        self.conn: dbapi.Connection


config = Config()


def setup_module(module: Any) -> None:
    config.conn = dbapi.connect(
        address=os.environ["HANA_DB_ADDRESS"],
        port=int(os.environ["HANA_DB_PORT"]),
        user=os.environ["HANA_DB_USER"],
        password=os.environ["HANA_DB_PASSWORD"],
    )


def teardown_module(module: Any) -> None:
    config.conn.close()


@pytest.fixture(scope="module")
def graph() -> HanaRdfGraph:
    return HanaRdfGraph(
        connection=config.conn,
        graph_uri="kgdocu_movies",
        auto_extract_ontology=True,
    )


@pytest.fixture(scope="module")
def llm() -> BaseChatModel:
    return init_llm(os.environ["AICORE_MODEL_ID"])


@pytest.fixture(scope="module")
def agent(graph: HanaRdfGraph, llm: BaseChatModel) -> Any:
    return HanaSparqlQAAgent.create_agent(graph=graph, model=llm)


@pytest.fixture(scope="module")
def trajectory_match_evaluator() -> Any:
    return create_trajectory_match_evaluator(
        trajectory_match_mode="superset",
        tool_args_match_mode="ignore",
    )


@pytest.fixture
def trajectory_llm_judge(llm: BaseChatModel) -> Any:
    return create_trajectory_llm_as_judge(
        prompt=TRAJECTORY_ACCURACY_PROMPT_WITH_REFERENCE,
        judge=llm,
    )


def _run_agent(agent: Any, input_messages: list[dict]) -> list[dict]:
    result = agent.invoke({"messages": input_messages})
    return result["messages"]


_ALL_TRAJECTORY_CASES = (
    [("trajectory", *case) for case in TRAJECTORY_EVALS]
    + [("off_topic", *case) for case in OFF_TOPIC_EVALS]
    + [("followup", *case) for case in FOLLOWUP_EVALS]
)


@pytest.mark.parametrize(
    "category,question,reference_trajectory,input_messages",
    _ALL_TRAJECTORY_CASES,
    ids=[f"{cat}::{q}" for cat, q, _, _ in _ALL_TRAJECTORY_CASES],
)
def test_trajectory_match(
    agent: HanaSparqlQAAgent,
    trajectory_match_evaluator: Any,
    category: str,
    question: str,
    reference_trajectory: list[dict],
    input_messages: list[dict],
) -> None:
    outputs = _run_agent(agent, input_messages)
    result = trajectory_match_evaluator(
        outputs=outputs,
        reference_outputs=reference_trajectory,
    )
    assert result["score"] is True, (
        f"[{category}] trajectory mismatch for {question!r}.\n"
        f"Reasoning: {result.get('comment')}\n"
        f"Got: {json.dumps(outputs, indent=2, default=str)}"
    )


@pytest.mark.parametrize(
    "category,question,reference_trajectory,input_messages",
    _ALL_TRAJECTORY_CASES,
    ids=[f"{cat}::{q}" for cat, q, _, _ in _ALL_TRAJECTORY_CASES],
)
def test_trajectory_llm_judge(
    agent: HanaSparqlQAAgent,
    trajectory_llm_judge: Any,
    category: str,
    question: str,
    reference_trajectory: list[dict],
    input_messages: list[dict],
) -> None:
    outputs = _run_agent(agent, input_messages)
    result = trajectory_llm_judge(
        outputs=outputs,
        reference_outputs=reference_trajectory,
    )
    assert result["score"] is True, (
        f"[{category}] LLM judge scored trajectory as inaccurate "
        f"for {question!r}.\nReasoning: {result.get('comment')}"
    )


GRADER_INSTRUCTIONS = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, and the \
STUDENT RESPONSE.

Here is the grade criteria to follow:
(1) Grade the student responses based ONLY on their factual accuracy relative \
to the ground truth answer.
(2) Ensure that the student response does not contain any conflicting \
statements.
(3) It is OK if the student response contains more information than the \
ground truth response, as long as it is factually accurate relative to the \
ground truth response.

Correctness:
True means that the student's response meets all of the criteria.
False means that the student's response does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and \
conclusion are correct."""


class Grade(TypedDict):
    reasoning: Annotated[
        str,
        ...,
        "Explain your reasoning for whether the actual response is correct or not.",
    ]
    is_correct: Annotated[
        bool,
        ...,
        "True if the student response is mostly or exactly correct, otherwise False.",
    ]


@pytest.fixture
def answer_grader(llm: BaseChatModel) -> Any:
    return llm.with_structured_output(Grade, method="json_schema", strict=True)


@pytest.mark.parametrize(
    "question,reference_answer",
    ANSWER_EVALS,
    ids=[q for q, _ in ANSWER_EVALS],
)
def test_answer_llm_judge(
    agent: Any,
    answer_grader: Any,
    question: str,
    reference_answer: str,
) -> None:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    agent_answer = result["messages"][-1].content
    assert agent_answer, f"Agent produced no final answer for {question!r}"

    user_message = (
        f"QUESTION: {question}\n"
        f"GROUND TRUTH RESPONSE: {reference_answer}\n"
        f"STUDENT RESPONSE: {agent_answer}"
    )
    grade: Grade = answer_grader.invoke(
        [
            {"role": "system", "content": GRADER_INSTRUCTIONS},
            {"role": "user", "content": user_message},
        ]
    )
    assert grade["is_correct"], (
        f"LLM judge rejected the agent's answer for {question!r}.\n"
        f"Reference: {reference_answer}\n"
        f"Agent:     {agent_answer}\n"
        f"Reasoning: {grade['reasoning']}"
    )
