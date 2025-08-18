LoRa for optimizing underlying LLM of DSPY agent for HotPotQA:

We have a DSPY React agent for HotPotQA with 3 tools:
1) Evaluate math expression
2) Search Wikipedia abstracts
3) Search web results

HotPotQA data from https://huggingface.co/datasets/hotpotqa/hotpot_qa

We want to:
0. Get training samples for our agent traces with Gemini Flash 2.5. What does the React agent with the smart LLM think we should do at each step?
1. Perform LoRa on Llama 3.2 3B with training samples. We need to train the dumb LLM to think like the smart one!
2. Compare accuracy of Dumb vs Smart vs Trained LLM!
