from langchain_core.language_models import BaseLanguageModel
from langchain_hana import HanaRdfGraph
class HanaSparqlQAAgent:
  """Agent for answering questions using SPARQL and HANA RDF Graphs"""

  SYSTEM_PROMPT = """
  """

  def __init__(
      self,
      graph: HanaRdfGraph,
      llm: BaseLanguageModel,
      system_prompt = SYSTEM_PROMPT,
  ):
    self.graph = graph
    self.llm = llm
    self.system_prompt = system_prompt
