import os

from dotenv import load_dotenv
import httpx
import dspy


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

    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": max_results,
        "include_answer": False,
        "include_raw_content": False,
        "include_images": False,
    }

    with httpx.Client(timeout=30) as client:
        response = client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        results = []
        for result in data.get("results", [])[:max_results]:
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
    "question -> answer: float", tools=[evaluate_math, search_wikipedia, search_web]
)

if __name__ == "__main__":
    load_dotenv(".env")

    # Configure DSPY to use our Ollama endpoint
    dspy.configure(lm=dspy.LM("gemini/gemini-2.5-flash-preview-05-20"), cache=False)

    # More challenging HotpotQA question requiring multi-hop reasoning
    pred = hotpotqa_agent(
        question="How many years passed between the founding of Harvard University and the birth year of Albert Einstein?"
    )

    print(f"Answer: {pred.answer}")
    dspy.inspect_history(n=10)
