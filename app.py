import llama_index
from llama_index.readers.file import CSVReader
from llama_index.readers.file import DocxReader
from llama_index.readers.file import EpubReader
from llama_index.readers.file import FlatReader
from llama_index.readers.file import HTMLTagReader
from llama_index.readers.file import HWPReader
from llama_index.readers.file import IPYNBReader
from llama_index.readers.file import ImageCaptionReader
from llama_index.readers.file import ImageReader
from llama_index.readers.file import ImageTabularChartReader
from llama_index.readers.file import ImageVisionLLMReader
from llama_index.readers.file import MarkdownReader
from llama_index.readers.file import MboxReader
from llama_index.readers.file import PDFReader
from llama_index.readers.file import PagedCSVReader
from llama_index.readers.file import PandasCSVReader
from llama_index.readers.file import PptxReader
from llama_index.readers.file import PyMuPDFReader
from llama_index.readers.file import RTFReader
from llama_index.readers.file import UnstructuredReader
from llama_index.readers.file import VideoAudioReader
from llama_index.readers.file import XMLReader
from llama_index.readers.chroma import ChromaReader
from llama_index.readers.web import AsyncWebPageReader
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.readers.web import KnowledgeBaseWebReader
from llama_index.readers.web import MainContentExtractorReader
from llama_index.readers.web import NewsArticleReader
from llama_index.readers.web import ReadabilityWebPageReader
from llama_index.readers.web import RssNewsReader
from llama_index.readers.web import RssReader
from llama_index.readers.web import SimpleWebPageReader
from llama_index.readers.web import SitemapReader
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.readers.web import UnstructuredURLLoader
from llama_index.readers.web import WholeSiteReader


##LlamaParse
import llama_parse
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
# parser = LlamaParse(
#     api_key="llx-...",  # can also be set in your env as LLAMA_CLOUD_API_KEY
#     result_type="markdown",  # "markdown" and "text" are available
#     num_workers=4, # if multiple files passed, split in `num_workers` API calls
#     verbose=True,
#     language="en" # Optionaly you can define a language, default=en
# )
# # sync
# documents = parser.load_data("./my_file.pdf")

# # sync batch
# documents = parser.load_data(["./my_file1.pdf", "./my_file2.pdf"])

# # async
# documents = await parser.aload_data("./my_file.pdf")

# # async batch
# documents = await parser.aload_data(["./my_file1.pdf", "./my_file2.pdf"])

from dspy.modules.anthropic import Claude
anthropicChat = Claude(model="claude-3-opus-20240229", port=ports, max_tokens=150)

def choose_reader(file_path: str) -> Any:
    """Choose the appropriate reader based on the file extension."""

    ext = os.path.splitext(file_path)[1].lower()
    
    readers: Dict[str, Any] = {
        ".json": JSONFileReader(),
        ".csv": CSVFileReader(),
        ".xlsx": ExcelSheetReader(),
        ".xls": ExcelSheetReader(),
        ".html": HTMLFileReader(),
        ".pdf": PDFMinerReader(),
        # Add more extensions and their corresponding readers as needed...
    }

    return readers.get(ext, None)

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Loads documents from files within a specified folder"""
    
    documents = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            full_path = os.path.join(root, filename)
            
            reader = choose_reader(full_path)

            if reader:
                print(f"Loading document from '{filename}' with {type(reader).__name__}")
                
                try:
                    docs = list(reader.load_data(input_files=[full_path]))
                    documents.extend(docs)
                    
                except Exception as e:
                    print(f"Failed to load document from '{filename}'. Error: {e}")
    # Convert to langchain format
    documents = [ doc.to_langchain_format()
    for doc in documents
    ]                       
    return documents

# Ragas : https://colab.research.google.com/gist/virattt/6a91d2a9dcf99604637e400d48d2a918/ragas-first-look.ipynb
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

# generator with openai models
generator = TestsetGenerator.with_openai()

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

# visualize the dataset as a pandas DataFrame
#dataframe = testset.to_pandas()
#dataframe.head(10)


#### DSPY APPLICATION LOGIC GOES HERE

## LOADING DATA
%load_ext autoreload
%autoreload 2

# %set_env CUDA_VISIBLE_DEVICES=7
# import sys; sys.path.append('/future/u/okhattab/repos/public/stanfordnlp/dspy')

import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dsp.modules.lm import LM

class Claude(LM):
    """Wrapper around anthropic's API. Supports both the Anthropic and Azure APIs."""
    def __init__(
            self,
            model: str = "claude-3-opus-20240229",
            api_key: Optional[str] = None,
            api_base: Optional[str] = None,
            **kwargs,
    ):
        super().__init__(model)

        try:
            from anthropic import Anthropic, RateLimitError
        except ImportError as err:
            raise ImportError("Claude requires `pip install anthropic`.") from err
        
        self.provider = "anthropic"
        self.api_key = api_key = os.environ.get("ANTHROPIC_API_KEY") if api_key is None else api_key
        self.api_base = BASE_URL if api_base is None else api_base

        self.kwargs = {
            "temperature": 0.0 if "temperature" not in kwargs else kwargs["temperature"],
            "max_tokens": min(kwargs.get("max_tokens", 4096), 4096),
            "top_p": 1.0 if "top_p" not in kwargs else kwargs["top_p"],
            "top_k": 1 if "top_k" not in kwargs else kwargs["top_k"],
            "n": kwargs.pop("n", kwargs.pop("num_generations", 1)),
            **kwargs,
        }
        self.kwargs["model"] = model
        self.history: list[dict[str, Any]] = []
        self.client = Anthropic(api_key=api_key)

ports = [7140, 7141, 7142, 7143, 7144, 7145]
#llamaChat = dspy.HFClientTGI(model="meta-llama/Llama-2-13b-chat-hf", port=ports, max_tokens=150) (DELETED)
colbertv2 = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

# Instantiate Claude with desired parameters
claude_model = Claude(model="claude-3-opus-20240229")

# Configure dspy settings with Claude as the language model
dspy.settings.configure(rm=colbertv2, lm=claude_model)
#dspy.settings.configure(rm=colbertv2, lm=llamaChat) #Llama change into model based on line 166

dataset = dataframe
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]
testset = [x.with_inputs('question') for x in dataset.test]

#len(trainset), len(devset), len(testset)
#trainset[0]

from dsp.utils.utils import deduplicate

#class BasicMH(dspy.Module):
#    def __init__(self, passages_per_hop=3):
#       super().__init__()

#        self.retrieve = dspy.Retrieve(k=passages_per_hop)
#        self.generate_query = [dspy.ChainOfThought("context, question -> search_query") for _ in range(2)]
#        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
#    def forward(self, question):
#        context = []
        
#        for hop in range(2):
#            search_query = self.generate_query[hop](context=context, question=question).search_query
#            passages = self.retrieve(search_query).passages
#            context = deduplicate(context + passages)

#        return self.generate_answer(context=context, question=question).copy(context=context)

class BasicMH(dspy.Module):
    def __init__(self, claude_model, passages_per_hop=3):
        super().__init__()

        self.claude_model = claude_model
        self.passages_per_hop = passages_per_hop
    
    def forward(self, question):
        context = []
        
        for hop in range(2):
            # Retrieval using Claude model
            search_results = self.claude_model.search(question, context=context, k=self.passages_per_hop)
            passages = [result.passage for result in search_results]
            context = deduplicate(context + passages)

        # Generation using Claude model
        answer = self.claude_model.generate(context=context, question=question)

        return answer

        
## Compiling using meta-llama/Llama-2-13b-chat-hf
#RECOMPILE_INTO_MODEL_FROM_SCRATCH = False
#NUM_THREADS = 24

#metric_EM = dspy.evaluate.answer_exact_match

#if RECOMPILE_INTO_MODEL_FROM_SCRATCH:
#    tp = BootstrapFewShotWithRandomSearch(metric=metric_EM, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
#    basicmh_bs = tp.compile(BasicMH(), trainset=trainset[:50], valset=trainset[50:200])

#    ensemble = [prog for *_, prog in basicmh_bs.candidate_programs[:4]]

#    for idx, prog in enumerate(ensemble):
#        # prog.save(f'multihop_llama213b_{idx}.json')
#        pass
#if not RECOMPILE_INTO_MODEL_FROM_SCRATCH:
#    ensemble = []

#    for idx in range(4):
#        prog = BasicMH()
#        prog.load(f'multihop_llama213b_{idx}.json')
#        ensemble.append(prog)
#llama_program = ensemble[0]
#RECOMPILE_INTO_MODEL_FROM_SCRATCH = False
#NUM_THREADS = 24

metric_EM = dspy.evaluate.answer_exact_match

if RECOMPILE_INTO_MODEL_FROM_SCRATCH:
    tp = BootstrapFewShotWithRandomSearch(metric=metric_EM, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
    # Compile the Claude model using BootstrapFewShotWithRandomSearch
    claude_bs = tp.compile(Claude(), trainset=trainset[:50], valset=trainset[50:200])

    # Get the compiled programs
    ensemble = [prog for *_, prog in claude_bs.candidate_programs[:4]]

    for idx, prog in enumerate(ensemble):
        # Save the compiled Claude models if needed
        # prog.save(f'multihop_llama213b_{idx}.json')
        pass
else:
    ensemble = []

    for idx in range(4):
        # Load the previously trained Claude models
        claude_model = Claude(model=f'multihop_claude3opus_{idx}.json') #need to prepare this .json file
        ensemble.append(claude_model)

# Select the first Claude model from the ensemble
claude_program = ensemble[0]





# LlamaPack example
from llama_index.core.llama_pack import download_llama_pack

# We will show you how to import the agent from these files!

# from llama_index.core.llama_pack import download_llama_pack

# # download and install dependencies
# download_llama_pack("LLMCompilerAgentPack", "./llm_compiler_agent_pack")
# From here, you can use the pack. You can import the relevant modules from the download folder (in the example below we assume it's a relative import or the directory has been added to your system path).

# # setup pack arguments

# from llama_index.core.agent import AgentRunner
# from llm_compiler_agent_pack.step import LLMCompilerAgentWorker

# agent_worker = LLMCompilerAgentWorker.from_tools(
#     tools, llm=llm, verbose=True, callback_manager=callback_manager
# )
# agent = AgentRunner(agent_worker, callback_manager=callback_manager)

# # start using the agent
# response = agent.chat("What is (121 * 3) + 42?")
# You can also use/initialize the pack directly.

# from llm_compiler_agent_pack.base import LLMCompilerAgentPack

# agent_pack = LLMCompilerAgentPack(tools, llm=llm)
# The run() function is a light wrapper around agent.chat().

# response = pack.run("Tell me about the population of Boston")
# You can also directly get modules from the pack.

# # use the agent
# agent = pack.agent
# response = agent.chat("task")


# from llama_parse import LlamaParse
# from llama_index.core import SimpleDirectoryReader

# parser = LlamaParse(
#     api_key="llx-...",  # can also be set in your env as LLAMA_CLOUD_API_KEY
#     result_type="markdown",  # "markdown" and "text" are available
#     verbose=True
# )

# file_extractor = {".pdf": parser}
# documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

