

def choose_reader(file_path: str) -> Any:
    ext = os.path.splitext(file_path)[1].lower()
    
    readers = {
        ".json": JSONFileReader(),
        ".csv": CSVReader(),
        ".docx": DocxReader(),
        ".epub": EpubReader(),
        ".flat": FlatReader(),  # This is an assumption; adjust based on actual reader
        ".html": HTMLTagReader(),
        ".hwp": HWPReader(),
        ".ipynb": IPYNBReader(),
        ".png": ImageReader(),  # Assuming generic image handling
        ".jpg": ImageReader(),
        ".jpeg": ImageReader(),
        # Continue for all file types...
    }


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

# Dummy backend function for handling user query
def handle_query(user_query: str) -> str:
    # Placeholder for processing user query
    return f"Processed query: {user_query}"

# Dummy backend function for handling repository input
def handle_repository(repository_link: str) -> str:
    # Placeholder for processing repository input
    return f"Processed repository link: {repository_link}"

# New dummy function for handling synthetic data generation
def handle_synthetic_data(schema_class_name: str, sample_size: int) -> str:
    # Placeholder for generating synthetic data based on the schema class name and sample size
    return f"Synthetic data for schema '{schema_class_name}' with {sample_size} samples has been generated."

# New dummy function for handling file uploads
def handle_file_upload(uploaded_file):
    # Placeholder for processing the uploaded file
    if uploaded_file is not None:
        return f"Uploaded file '{uploaded_file.name}' has been processed."
    return "No file was uploaded."
    
from dspy.modules.anthropic import Claude
anthropicChat = Claude(model="claude-3-opus-20240229", port=ports, max_tokens=150)




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
# LlamaPack example
# from llama_index.core.llama_pack import download_llama_pack

#Ragas : https://colab.research.google.com/gist/virattt/6a91d2a9dcf99604637e400d48d2a918/ragas-first-look.ipynb
#from ragas.testset.generator import TestsetGenerator
#from ragas.testset.evolutions import simple, reasoning, multi_context

# generator with openai models
# generator = TestsetGenerator.with_openai()

# generate testset
#testset = generator.generate_with_langchain_docs(documents, test_size=10, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25})

# visualize the dataset as a pandas DataFrame
#dataframe = testset.to_pandas()
#dataframe.head(10)


#### DSPY APPLICATION LOGIC GOES HERE


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