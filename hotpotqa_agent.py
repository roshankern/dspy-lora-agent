import os

from dotenv import load_dotenv

from tavily import TavilyClient
import dspy
from dspy.datasets import HotPotQA


# Test the endpoint with a simple math problem
def evaluate_math(expression: str):
    """Evaluate a mathematical expression safely

    Args:
        expression (str): The mathematical expression to evaluate

    Returns:
        float: The result of the evaluation or an error message
    """

    try:
        # Only allow safe mathematical operations
        allowed_names = {
            "__builtins__": {},
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "pow": pow,
        }
        # Add math functions
        import math

        for name in dir(math):
            if not name.startswith("_"):
                allowed_names[name] = getattr(math, name)

        result = eval(expression, allowed_names)
        return float(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


def search_wikipedia(query: str):
    """Search Wikipedia abstracts with a given query

    Args:
        query (str): The search query

    Returns:
        List[str]: A list of Wikipedia article abstracts
    """
    results = dspy.ColBERTv2(url="http://20.102.90.50:2017/wiki17_abstracts")(
        query, k=3
    )
    return [x["text"] for x in results]


def search_web(query: str, max_results: int):
    """
    Performs a web search using the Tavily API and returns a list of search results.

    Args:
        query (str): The search query string.
        max_results (int): The maximum number of search results to return.

    Returns:
        list[dict]: A list of dictionaries, each containing:
            - 'title' (str): The title of the search result.
            - 'snippet' (str): A snippet or content summary of the result.
            - 'url' (str): The URL of the search result (truncated to 50 characters).
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    tavily_client = TavilyClient(api_key=api_key)
    response = tavily_client.search(
        query,
        max_results=max_results,
        search_depth="basic",
        include_answer=False,
        include_raw_content=False,
        include_images=False,
    )

    results = []
    for result in response.get("results", [])[:max_results]:
        snippet = result.get("content", "")
        title = result.get("title", "No title")
        results.append(
            {
                "title": title,
                "snippet": snippet,
                "url": result.get("url", "")[:50],
            }
        )

    return results


hotpotqa_agent = dspy.ReAct(
    "question -> answer",
    tools=[evaluate_math, search_wikipedia],  # , search_web],
    max_iters=5,
)

if __name__ == "__main__":
    load_dotenv(".env")

    # Configure DSPY with a language model
    # lm = dspy.LM(
    #     "openai/llama3.2:3b",  # anything smaller than 3.2:3b is too dumb for DSPY parsing
    #     api_key="makora_bio_endpoint",
    #     api_base="https://roshan-kern--ollama-endpoint-ollama-api.modal.run/v1",
    #     cache=False,
    # )
    lm = dspy.LM(
        "openai/unsloth/Llama-3.2-3B-Instruct",
        api_key="",
        api_base="https://roshan-kern--hf-endpoint-serve-original-model.modal.run/v1",
        cache=False,
    )
    lm = dspy.LM(
        "openai/rshn-krn/hotpotqa-agent-sft-llm",
        api_key="",
        api_base="https://roshan-kern--hf-endpoint-serve-sft-model.modal.run/v1",
        cache=False,
    )
    lm = dspy.LM("gemini/gemini-2.5-flash-preview-05-20", cache=False)
    dspy.configure(lm=lm, temperature=0.7)

    # More challenging HotpotQA question requiring multi-hop reasoning
    pred = hotpotqa_agent(
        question="What was the tile of Björk's third album, released in 1997 and featuring the single All Is Full Of Love?"
    )

    print(f"Answer: {pred.answer}")
    dspy.inspect_history(n=10)
