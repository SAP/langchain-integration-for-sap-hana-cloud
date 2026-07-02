from typing import Any, Callable, Sequence

from langchain.agents import create_agent as create_base_agent
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import BaseTool, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage

from langchain_hana.graphs import HanaRdfGraph

from .prompts import SYSTEM_PROMPT


class HanaSparqlQAAgent:
    """Agent for answering questions using SPARQL against a SAP HANA Cloud RDF graph.

    The agent is backed by LangChain's `createAgent` harness and, by default, is
    equipped with two tools:

    1. `retrieveOntology`- returns the serialized ontology of the graph.
    2. `executeSparql`- runs a SPARQL query against the graph and returns the result.

    Example:
    ```python
        agent = HanaSparqlQAAgent.create_agent(
            graph=graph,
            model=model,
        )
        response = agent.invoke(query)
    ```

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        graph: HanaRdfGraph,
        tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None,
        middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None,
        system_prompt: str | SystemMessage | None,
        include_default_tools: bool,
        include_default_middleware: bool,
    ):
        self.graph = graph
        self.ontology = self.graph.get_schema.serialize(format="turtle")

        self.system_prompt: str | SystemMessage
        if system_prompt is None:
            self.system_prompt = SYSTEM_PROMPT.format(
                from_clause=self.graph.from_clause
            )
        else:
            self.system_prompt = system_prompt

        # Create tools bound to this instance
        self.tools: list[BaseTool | Callable[..., Any] | dict[str, Any]]
        if tools:
            self.tools = list(tools)
        else:
            self.tools = []

        if include_default_tools:
            self.tools.extend(
                [self._create_ontology_tool(), self._create_sparql_tool()]
            )

        # Create the middleware
        self.middleware: list[AgentMiddleware[Any, Any, Any]]
        if middleware:
            self.middleware = list(middleware)
        else:
            self.middleware = []

        if include_default_middleware:
            self.middleware.append(ModelRetryMiddleware(max_retries=3))
            self.middleware.append(ToolRetryMiddleware(max_retries=2))

    def _create_ontology_tool(self) -> BaseTool:
        """Creates the tool that returns the ontology of the HANA RDF graph"""

        @tool
        def retrieve_ontology() -> str:
            """Retrieve ontology from the HANA RDF Graph"""
            return f"Ontology Information:\n{self.ontology}"

        return retrieve_ontology

    def _create_sparql_tool(self) -> BaseTool:
        """Creates the tool that executes a SPARQL query on the HANA RDF graph"""

        @tool
        def execute_sparql(query: str) -> str:
            """Query the HANA RDF graph and return the fetched triples as a string.
            Args:
                query: SPARQL query to execute on the RDF graph
            """
            try:
                query_result = self.graph.query(query)
            except Exception as e:
                return f"Error executing SPARQL query: {e}"
            return f"SPARQL Query Result:\n{query_result}"

        return execute_sparql

    @classmethod
    def create_agent(
        cls,
        graph: HanaRdfGraph,
        model: str | BaseChatModel,
        tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
        system_prompt: str | SystemMessage | None = None,
        middleware: Sequence[AgentMiddleware[Any, Any, Any]] | None = None,
        include_default_tools: bool = True,
        include_default_middleware: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Create a new SPARQL QA agent instance

        Args:
            graph: The HANA RDF graph the agent queries.
            model: Language model to use for the agent.
            tools: Optional additional tools to expose to the agent.
            system_prompt: Optional system prompt for the agent.
                Defaults to a built-in prompt.
            middleware: Optional list of middleware to include in the agent.
            include_default_tools: Whether to include default tools.
                Defaults to True.
            include_default_middleware: Whether to include default
                middleware. Defaults to True.
            **kwargs: Additional keyword arguments for the base agent creation.

        Returns:
            A new SPARQL QA agent instance.
        """

        instance = cls(
            graph=graph,
            tools=tools,
            middleware=middleware,
            system_prompt=system_prompt,
            include_default_tools=include_default_tools,
            include_default_middleware=include_default_middleware,
        )
        return create_base_agent(
            model,
            tools=instance.tools,
            system_prompt=instance.system_prompt,
            middleware=instance.middleware,
            **kwargs,
        )
