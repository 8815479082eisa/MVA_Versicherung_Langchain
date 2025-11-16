from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI # Example LLM, can be replaced
from langchain_core.documents import Document
from typing import List

class AnswerGenerator:
    def __init__(self, llm=None, temperature=0.7):
        self.llm = llm if llm else ChatOpenAI(model="gpt-4o-mini", temperature=temperature)
        self.prompt_template = self._create_prompt_template()
        self.output_parser = StrOutputParser()

    def _create_prompt_template(self) -> ChatPromptTemplate:
        # Exemplarischer Prompt-Header aus der MVA
        template = (
            "SYSTEM: Du bist ein fachlicher Assistent für Versicherungsbedingungen. "
            "Antworte nur auf Basis der bereitgestellten Passagen. "
            "Zitiere Quelle(n) mit [Dok-ID:Seite/Abschnitt]. Wenn unsicher: 'Keine gesicherte Auskunft, bitte Rückfrage'.\n\n"
            "USER: {question}\n"
            "CONTEXT: {context}\n\n"
            "TASK: Erzeuge eine präzise, kurze Antwort + Quellenblock."
        )
        return ChatPromptTemplate.from_template(template)

    def format_docs(self, docs: List[Document]) -> str:
        """Formats the retrieved documents into a string for the LLM context."""
        formatted_string = ""
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "unknown")
            # Assuming 'source' in metadata refers to Dok-ID, and 'page' refers to Seite/Abschnitt
            formatted_string += f"Passage {i+1} [Dok-ID:{source}:Seite/{page}]:\n{doc.page_content}\n\n"
        return formatted_string.strip()

    def generate_answer(self, question: str, documents: List[Document]) -> str:
        """Generates an answer based on the question and retrieved documents."""
        context_string = self.format_docs(documents)

        rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        # Langchain Runnable expects `context` to be string, `question` to be string.
        # This may need adjustment depending on how the upstream pipeline passes data.
        # For now, we will pass the formatted context directly.
        print(f"Generating answer for question: '{question}'")
        print("Using context from retrieved documents...")

        # The .invoke() method expects a dictionary with keys matching prompt template variables.
        # In our case, the prompt expects 'question' and 'context'.
        response = rag_chain.invoke({"question": question, "context": context_string})
        print("Answer generation complete.")
        return response

if __name__ == "__main__":
    # Example Usage:
    # Create a dummy LLM and some dummy documents for testing
    class DummyLLM:
        def invoke(self, prompt_value):
            # Simulate LLM response with citations
            if "fire damage" in prompt_value.lower() and "policy_A" in prompt_value:
                return "The deductible for fire damage is $500 [Dok-ID:policy_A:Seite/1]."
            elif "theft" in prompt_value.lower() and "policy_B" in prompt_value:
                return "The insurance covers theft incidents [Dok-ID:policy_B:Seite/2]."
            return "Keine gesicherte Auskunft, bitte Rückfrage."

    dummy_llm = DummyLLM()
    answer_generator = AnswerGenerator(llm=dummy_llm)

    dummy_docs = [
        Document(page_content="The deductible for fire damage is $500.", metadata={"source": "policy_A", "page": 1}),
        Document(page_content="Insurance covers theft and fire incidents.", metadata={"source": "policy_B", "page": 2}),
        Document(page_content="General policy terms and conditions.", metadata={"source": "policy_C", "page": 3}),
    ]

    question1 = "What is the deductible for fire damage?"
    answer1 = answer_generator.generate_answer(question1, dummy_docs)
    print(f"\nQuestion: {question1}")
    print(f"Answer: {answer1}\n")

    question2 = "Does the policy cover theft?"
    answer2 = answer_generator.generate_answer(question2, dummy_docs)
    print(f"Question: {question2}")
    print(f"Answer: {answer2}\n")

    question3 = "What about hail damage?"
    answer3 = answer_generator.generate_answer(question3, dummy_docs[:1]) # Only one doc, not enough info
    print(f"Question: {question3}")
    print(f"Answer: {answer3}\n")
