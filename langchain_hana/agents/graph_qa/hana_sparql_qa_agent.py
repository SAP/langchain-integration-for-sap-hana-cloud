from langchain_core.language_models import BaseLanguageModel
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_hana import HanaRdfGraph
from langgraph.prebuilt import ToolNode
from .prompts import SYSTEM_PROMPT

class HanaSparqlQAAgent:
    """Agent for answering questions using SPARQL and HANA RDF Graphs"""

    def __init__(self, graph: HanaRdfGraph, tools=None, middleware=None, system_prompt: str = None, include_default_tools: bool = True, include_default_middleware: bool = True):
        self.graph = graph
        self.ontology = self.graph.get_schema.serialize(format="turtle")
        self.system_prompt = system_prompt or SYSTEM_PROMPT.format(self.graph.from_clause)

        # Create tools bound to this instance
        if tools:
            self.tools = tools
        else:
            self.tools =[]
        
        if include_default_tools:
            self.tools.extend([self._create_ontology_tool(), self._create_sparql_tool()])

        # Create the middleware
        if middleware:
          self.middleware = middleware
        else:
          self.middleware = []

        if include_default_middleware:
          self.middleware.append(ModelCallLimitMiddleware(run_limit=10))

    def _create_ontology_tool(self):
        @tool
        def retrieve_ontology() -> str:
            """Retrieve ontology from the HANA RDF Graph"""
            return f"Ontology Information:\n{self.ontology}"
        return retrieve_ontology
    
    def _create_sparql_tool(self):
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
    def create_agent(cls, graph: HanaRdfGraph, model: BaseLanguageModel, tools=None, system_prompt: str = None, middleware=None, **kwargs):
        """Create a new SPARQL QA agent instance"""
        instance = cls(graph=graph, tools=tools, middleware=middleware, system_prompt=system_prompt)
        return create_agent(model, tools=instance.tools, system_prompt=instance.system_prompt, middleware=instance.middleware, **kwargs)
