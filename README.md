# Adapt-a-RAG: Adaptable Retrieval Augmented Generation

Adapt-a-RAG is an adaptable retrieval augmented application that provides question answering over documents, GitHub repositories, and websites. It takes data, creates synthetic data, and uses that synthetic data to optimize the prompts of the Adapt-a-RAG application. The application recompiles itself every run in a unique and adapted way to the user query.

## Table of Contents

- [Introduction](#introduction)
- [Setup](#setup)
- [How It Works](#how-it-works)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Adapt-a-RAG is an innovative application that leverages the power of retrieval augmented generation to provide accurate and relevant answers to user queries. By adapting itself to each query, Adapt-a-RAG ensures that the generated responses are tailored to the specific needs of the user.

The application utilizes various data sources, including documents, GitHub repositories, and websites, to gather information and generate synthetic data. This synthetic data is then used to optimize the prompts of the Adapt-a-RAG application, enabling it to provide more accurate and contextually relevant answers.

## Setup

To set up Adapt-a-RAG, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/Josephrp/adapt-a-rag.git
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure the necessary API keys and environment variables.

4. Run the application:
   ```
   python main.py
   ```

## How It Works

Adapt-a-RAG works by following these key steps:

1. **Data Collection**: The application collects data from various sources, including documents, GitHub repositories, and websites. It utilizes different reader classes such as `CSVReader`, `DocxReader`, `PDFReader`, `ChromaReader`, and `SimpleWebPageReader` to extract information from these sources.

2. **Synthetic Data Generation**: Adapt-a-RAG generates synthetic data using the collected data. It employs techniques such as data augmentation and synthesis to create additional training examples that can help improve the performance of the application.

3. **Prompt Optimization**: The synthetic data is used to optimize the prompts of the Adapt-a-RAG application. By fine-tuning the prompts based on the generated data, the application can generate more accurate and relevant responses to user queries.

4. **Recompilation**: Adapt-a-RAG recompiles itself every run based on the optimized prompts and the specific user query. This dynamic recompilation allows the application to adapt and provide tailored responses to each query.

5. **Question Answering**: Once recompiled, Adapt-a-RAG takes the user query and retrieves relevant information from the collected data sources. It then generates a response using the optimized prompts and the retrieved information, providing accurate and contextually relevant answers to the user.

## Contributing

We welcome contributions to Adapt-a-RAG! If you'd like to contribute, please follow these steps:

1. Fork the repository on GitHub.

2. Create a new branch from the `devbranch`:
   ```
   git checkout -b feature/your-feature-name devbranch
   ```

3. Make your changes and commit them with descriptive commit messages.

4. Push your changes to your forked repository:
   ```
   git push origin feature/your-feature-name
   ```

5. Open a pull request against the `devbranch` of the main repository.

Please ensure that your contributions adhere to the project's coding conventions and include appropriate tests and documentation.

## License

Adapt-a-RAG is released under the MIT License. See the [LICENSE](LICENSE) file for more details.