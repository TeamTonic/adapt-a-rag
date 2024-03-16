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
####LlamaParse
import llama_parse
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import random
from typing import List, Optional
from pydantic import BaseModel
import dspy
import gradio as gr
import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dsp.modules.lm import LM
from dsp.utils.utils import deduplicate
import os
import dotenv
from dotenv import load_dotenv, set_key
from pathlib import Path

# Assume all necessary imports for llama_index readers are correctly done at the beginning
def load_data_from_source(source: Union[str, dict]) -> Any:
    """
    Loads data from various sources using the appropriate llama_index reader based on the source type.

    :param source: A string representing a file path or a URL, or a dictionary specifying web content to fetch.
    :return: Loaded data.
    """
    if isinstance(source, str):
        print("Source is a string.")
        ext = os.path.splitext(source)[-1].lower()
        print(f"Detected extension: {ext}")

        if ext == '.csv':
            reader = CSVReader()
        elif ext == '.docx':
            reader = DocxReader()
        elif ext == '.epub':
            reader = EpubReader()
        elif ext == '.html':
            reader = HTMLTagReader()
        elif ext == '.hwp':
            reader = HWPReader()
        elif ext == '.ipynb':
            reader = IPYNBReader()
        elif ext in ['.png', '.jpg', '.jpeg']:
            reader = ImageReader()  # Assuming ImageReader can handle common image formats
        elif ext == '.md':
            reader = MarkdownReader()
        elif ext == '.mbox':
            reader = MboxReader()
        elif ext == '.pdf':
            reader = PDFReader()
        elif ext == '.pptx':
            reader = PptxReader()
        elif ext == '.rtf':
            reader = RTFReader()
        elif ext == '.xml':
            reader = XMLReader()
         elif source.startswith('http'):
            print("Source is a URL.")
            reader = AsyncWebPageReader()  # Simplified assumption for URLs
        else:
            print(f"Unsupported source type: {source}")
            raise ValueError(f"Unsupported source type: {source}")
    elif isinstance(source, dict):
        print("Source is a dictionary.")
        reader = AsyncWebPageReader()
    else:
        print("Source type is neither string nor dictionary.")
        raise TypeError("Source must be a string or dictionary.")
    
    print("Using reader to load data...")
    # Use the reader to load data
    data = reader.read(source)  # Adjust method name as necessary
    print("Data loaded successfully.")
    
    return data

def set_api_keys(anthropic_api_key: str, openai_api_key: str):
    """
    Function to securely set API keys by updating the .env file in the application's directory.
    This approach ensures that sensitive information is not hard-coded into the application.
    """
    print("Setting API keys...")
    # Define the path to the .env file
    env_path = Path('.') / '.env'
    
    print(f"Loading existing .env file from: {env_path}")
    # Load existing .env file or create one if it doesn't exist
    load_dotenv(dotenv_path=env_path, override=True)
    
    print("Updating .env file with new API keys...")
    # Update the .env file with the new values
    set_key(env_path, "ANTHROPIC_API_KEY", anthropic_api_key)
    set_key(env_path, "OPENAI_API_KEY", openai_api_key)
    
    print("API keys updated successfully.")
    # Returns a confirmation without exposing the keys
    return "API keys updated successfully in .env file. Please proceed with your operations."

def load_api_keys_and_prompts():
    """
    Loads API keys and prompts from an existing .env file into the application's environment.
    This function assumes the .env file is located in the same directory as the script.
    """
    print("Loading API keys and prompts...")
    # Define the path to the .env file
    env_path = Path('.') / '.env'
    
    print(f"Loading .env file from: {env_path}")
    # Load the .env file
    load_dotenv(dotenv_path=env_path)
    
    print("Accessing variables from the environment...")
    # Access the variables from the environment
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    field_prompt = os.getenv("FIELDPROMPT")
    example_prompt = os.getenv("EXAMPLEPROMPT")
    description_prompt = os.getenv("DESCRIPTIONPROMPT")
    
    print("API keys and prompts loaded successfully.")
    # Optionally, print a confirmation or return the loaded values
    return {
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "OPENAI_API_KEY": openai_api_key,
        "FIELDPROMPT": field_prompt,
        "EXAMPLEPROMPT": example_prompt,
        "DESCRIPTIONPROMPT": description_prompt
    }

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """Loads documents from files within a specified folder"""
    folder_path = "./add_your_files_here"
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

### DSPY DATA GENERATOR

class descriptionSignature(dspy.Signature):
  # add self.env.prompts
  field_name = dspy.InputField(desc=field_prompt)
  example = dspy.InputField(desc=example_prompt)
  description = dspy.OutputField(desc=description_prompt)

class SyntheticDataGenerator:
    def __init__(self, schema_class: Optional[BaseModel] = None, examples: Optional[List[dspy.Example]] = None):
        self.schema_class = schema_class
        self.examples = examples
        print("SyntheticDataGenerator initialized.")


    def generate(self, sample_size: int) -> List[dspy.Example]:
        print(f"Starting data generation for sample size: {sample_size}")
        if not self.schema_class and not self.examples:
            raise ValueError("Either a schema_class or examples must be provided.")
        if self.examples and len(self.examples) >= sample_size:
            print("No additional data generation needed.")
            return self.examples[:sample_size]

        additional_samples_needed = sample_size - (len(self.examples) if self.examples else 0)
        print(f"Generating {additional_samples_needed} additional samples.")
        generated_examples = self._generate_additional_examples(additional_samples_needed)

        return self.examples + generated_examples if self.examples else generated_examples

    def _define_or_infer_fields(self):
        print("Defining or inferring fields for data generation.")
        if self.schema_class:
            data_schema = self.schema_class.model_json_schema()
            properties = data_schema['properties']
        elif self.examples:
            inferred_schema = self.examples[0].__dict__['_store']
            descriptor = dspy.Predict(descriptionSignature)
            properties = {field: {'description': str((descriptor(field_name=field, example=str(inferred_schema[field]))).description)}
                          for field in inferred_schema.keys()}
        else:
            properties = {}
        return properties

    def _generate_additional_examples(self, additional_samples_needed: int) -> List[dspy.Example]:
        print(f"Generating {additional_samples_needed} additional examples.")
        properties = self._define_or_infer_fields()
        class_name = f"{self.schema_class.__name__ if self.schema_class else 'Inferred'}Signature"
        fields = self._prepare_fields(properties)

        signature_class = type(class_name, (dspy.Signature,), fields)
        generator = dspy.Predict(signature_class, n=additional_samples_needed)
        response = generator(sindex=str(random.randint(1, additional_samples_needed)))

        return [dspy.Example({field_name: getattr(completion, field_name) for field_name in properties.keys()})
                for completion in response.completions]

    def _prepare_fields(self, properties) -> dict:
        print("Preparing fields for the signature class.")
        return {
            '__doc__': f"Generates the following outputs: {{{', '.join(properties.keys())}}}.",
            'sindex': dspy.InputField(desc="a random string"),
            **{field_name: dspy.OutputField(desc=properties[field_name].get('description', 'No description'))
               for field_name in properties.keys()},
        }

# Generating synthetic data via existing examples
generator = SyntheticDataGenerator(examples=existing_examples)
dataframe = generator.generate(sample_size=5)

## LOADING DATA
%load_ext autoreload
%autoreload 2

# %set_env CUDA_VISIBLE_DEVICES=7
# import sys; sys.path.append('/future/u/okhattab/repos/public/stanfordnlp/dspy')

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

def main():
    with gr.Blocks() as demo:
        gr.Markdown("### Securely Input API Keys")
        with gr.Row():
            anthropic_api_key_input = gr.Textbox(label="Anthropic API Key", placeholder="Enter your Anthropic API Key", type="password")
            openai_api_key_input = gr.Textbox(label="OpenAI API Key", placeholder="Enter your OpenAI API Key", type="password")
        submit_button = gr.Button("Submit")
        confirmation_output = gr.Textbox(label="Confirmation", visible=False)  # Keep invisible for added security

        submit_button.click(
            fn=set_api_keys,
            inputs=[anthropic_api_key_input, openai_api_key_input],
            outputs=confirmation_output
        )

        with gr.Tab("User Query"):
            with gr.Row():
                user_query_input = gr.Textbox(label="Enter your query/prompt")
            query_button = gr.Button("Submit Query")
            query_output = gr.Textbox()

            query_button.click(
                fn=handle_query,
                inputs=[user_query_input],
                outputs=query_output
            )

        with gr.Tab("Repository Input"):
            with gr.Row():
                repository_link_input = gr.Textbox(label="Enter repository link")
            repository_button = gr.Button("Process Repository")
            repository_output = gr.Textbox()

            repository_button.click(
                fn=handle_repository,
                inputs=[repository_link_input],
                outputs=repository_output
            )

        with gr.Tab("Generate Synthetic Data"):
            with gr.Row():
                schema_input = gr.Textbox(label="Schema Class Name")
                sample_size_input = gr.Number(label="Sample Size", value=100)
            synthetic_data_button = gr.Button("Generate Synthetic Data")
            synthetic_data_output = gr.Textbox()

            synthetic_data_button.click(
                fn=handle_synthetic_data,
                inputs=[schema_input, sample_size_input],
                outputs=synthetic_data_output
            )

        with gr.Tab("Process Data"):
            with gr.Row():
                file_upload = gr.File(label="Upload Data File")
            file_upload_button = gr.Button("Process Uploaded File")
            file_upload_output = gr.Textbox()

            file_upload_button.click(
                fn=handle_file_upload,
                inputs=[file_upload],
                outputs=file_upload_output
            )

    demo.launch()

if __name__ == "__main__":
    main()