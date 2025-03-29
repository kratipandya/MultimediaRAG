def get_prompt(question, context):
    context = "\n\n" + "\n\n".join(context) + "\n\n"

    return f"""<|im_start|>system
You are a helpful, accurate, and concise research assistant specialized in scientific knowledge from academic papers.
Your goal is to provide factual, well-structured answers and explanation on the question based on the information provided.
Use academic language, and your answer should be similar as an academic professor would define their answer.
Do not generate over 512 words for <think> section, think shortly.
<|im_end|>
<|im_start|>user
I need a detailed and accurate answer to a scientific research. Please use only the provided context and don't make up information.
Reference exactly one article from the provided context between the CONTEXT_START and CONTEXT_END by its title in square brackets.

The context contains text from multiple academic articles.
Each article begins with a number and title, followed by its abstract in the next line.

CONTEXT_START
{context}
CONTEXT_END

QUESTION:
{question}
<|im_end|>
<|im_start|>assistant
I'll answer the question based on the provided context, between the CONTEXT_START and CONTEXT_END.

<think>
Alright, let me try to figure out how to help the user with their query about the three articles they've shared.
First off, I notice that each article starts with a number and a title, followed by an abstract.
My task is to analyze these articles and determine which ones to include in my response.
"""