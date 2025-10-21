"""Question answering over a SAP HANA graph using SPARQL."""

from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.callbacks.manager import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from langchain_hana.graphs import HanaRdfGraph

from .prompts import (
    SPARQL_GENERATION_SELECT_PROMPT,
    SPARQL_QA_PROMPT,
)


class HanaSparqlQAChain(BaseModel):
    """Chain for question-answering against a SAP HANA CLOUD Knowledge Graph Engine
    by generating SPARQL statements.

    Example:
        chain = HanaSparqlQAChain.from_llm(
            llm=llm,
            verbose=True,
            allow_dangerous_requests=True,
            graph=graph)
        response = chain.invoke({"query": "What is the population?"})

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    model_config = {"arbitrary_types_allowed": True}

    graph: HanaRdfGraph
    sparql_generation_chain: Runnable[Dict[str, Any], str]
    qa_chain: Runnable[Dict[str, Any], str]
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    allow_dangerous_requests: bool = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if self.allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database."
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        sparql_generation_prompt: BasePromptTemplate = SPARQL_GENERATION_SELECT_PROMPT,
        qa_prompt: BasePromptTemplate = SPARQL_QA_PROMPT,
        **kwargs: Any,
    ) -> HanaSparqlQAChain:
        sparql_generation_chain = sparql_generation_prompt | llm | StrOutputParser()
        qa_chain = qa_prompt | llm | StrOutputParser()
        return cls(
            qa_chain=qa_chain,
            sparql_generation_chain=sparql_generation_chain,
            **kwargs,
        )

    @staticmethod
    def extract_sparql(query: str) -> str:
        """Extract SPARQL code from a text.

        Args:
            query: Text to extract SPARQL code from.

        Returns:
            SPARQL code extracted from the text.
        """
        query = query.strip()
        querytoks = query.split("```")
        if len(querytoks) == 3:
            query = querytoks[1]
            if query.startswith("sparql"):
                query = query[6:]
        elif query.startswith("<sparql>") and query.endswith("</sparql>"):
            query = query[8:-9]
        return query

    def _ensure_common_prefixes(self, query: str) -> str:
        """
        Ensure common prefixes (rdf, rdfs, owl, xsd) are declared if used in the query.
        """
        common = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
        }
        # detect existing prefix declarations
        present = set()
        for line in query.splitlines():
            line_strip = line.strip()
            if line_strip.upper().startswith("PREFIX "):
                parts = line_strip.split()
                if len(parts) >= 2 and parts[1].endswith(":"):
                    present.add(parts[1][:-1])
        # build missing declarations
        missing = [(p, uri) for p, uri in common.items() if p not in present]
        if missing:
            prefix_lines = ""
            for p, uri in missing:
                if p in query:
                    prefix_lines += f"PREFIX {p}: <{uri}>\n"
            query = prefix_lines + query
        return query

    def invoke(
        self,
        inputs: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
    ) -> Dict[str, str]:
        """Generate SPARQL query, use it to look up in the graph and answer the question."""
        # Create callback manager (preserved from original)
        _run_manager = CallbackManagerForChainRun.get_noop_manager()
        if config and config.get("callbacks"):
            # Use the provided callbacks
            for callback in config["callbacks"]:
                if hasattr(callback, 'on_text'):
                    _run_manager = callback

        # Extract user question
        question = inputs[self.input_key]

        # Generate SPARQL query from the question and schema
        generated_sparql = self.sparql_generation_chain.invoke(
            {"prompt": question, "schema": self.graph.get_schema}, config=config
        )

        # Log the generated SPARQL
        _run_manager.on_text("Generated SPARQL:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_sparql, color="green", end="\n", verbose=self.verbose
        )

        # Extract the SPARQL code from the generated text and inject the from clause
        generated_sparql = self.extract_sparql(generated_sparql)
        generated_sparql = self.graph.inject_from_clause(generated_sparql)
        generated_sparql = self._ensure_common_prefixes(generated_sparql)
        _run_manager.on_text("Final SPARQL:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            generated_sparql, color="yellow", end="\n", verbose=self.verbose
        )

        # Execute the generated SPARQL query against the graph
        context = self.graph.query(generated_sparql, inject_from_clause=False)

        # Log the full context (SPARQL results)
        _run_manager.on_text("Full Context:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(context), color="green", end="\n", verbose=self.verbose
        )

        # Pass the question and query results into the QA chain
        result = self.qa_chain.invoke(
            {"prompt": question, "context": context}, config=config
        )

        # Return the final answer
        return {self.output_key: result}
