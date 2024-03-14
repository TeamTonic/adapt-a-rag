## from weaviate/recipes

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("question, contexts -> answer")
    
    def forward(self, question):
        contexts = self.retrieve(question).passages
        prediction = self.generate_answer(question=question, contexts=contexts
        return dspy.Prediction(answer=prediction.answer)

class Reranker(dspy.Signature):
    """Please rerank these documents."""
    
    context = dspy.InputField(desc="documents coarsely determined to be relevant to the question.")
    question = dspy.InputField()
    ranked_context = dspy.OutputField(desc="A ranking of documents by relevance to the question.")

class RAGwithReranker(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=5)
        self.reranker = dspy.ChainOfThought(Reranker)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        context = self.reranker(context=context, question=question).ranked_context
        pred = self.generate_answer(context=context, question=question).best_answer
        return dspy.Prediction(answer=pred)

class Summarizer(dspy.Signature):
    """Please summarize all relevant information in the context."""
    
    context = dspy.InputField(desc="documents determined to be relevant to the question.")
    question = dspy.InputField()
    summarized_context = dspy.OutputField(desc="A summarization of information in the documents that will help answer the quesetion.")

class RAGwithSummarizer(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.retrieve = dspy.Retrieve(k=5)
        self.summarizer = dspy.ChainOfThought(Summarizer)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        context = self.summarizer(context=context, question=question).summarized_context
        pred = self.generate_answer(context=context, question=question).best_answer
        return dspy.Prediction(answer=pred)

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()
import dspy
from dsp.utils import deduplicate

class MultiHopRAG(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_question = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_question[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.best_answer)

class MultiHopRAGwithSummarization(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()

        self.generate_question = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.summarizer = dspy.ChainOfThought(Summarizer)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        
        for hop in range(self.max_hops):
            query = self.generate_question[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            summarized_passages = self.summarizer(question=query, context=passages).summarized_context
            context.append(summarized_passages)

        pred = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=pred.best_answer)