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
    # documents = [ doc.to_langchain_format()
    # for doc in documents
    # ]                       
    return documents

# Ragas : https://colab.research.google.com/gist/virattt/6a91d2a9dcf99604637e400d48d2a918/ragas-first-look.ipynb
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
import os

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "InputOpenAIAPIHere"
# generator with openai models
generator = TestsetGenerator.with_openai()
# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

# visualize the dataset as a pandas DataFrame
#dataframe = testset.to_pandas()
#dataframe.head(10)


#### DSPY APPLICATION LOGIC GOES HERE


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

