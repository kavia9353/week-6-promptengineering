import pandas as pd
import os

# Create dataset folder if it does not exist
os.makedirs("dataset", exist_ok=True)

# 1. Prompts
prompts = [
    {
        "technique": "Zero-Shot",
        "prompt": "Explain artificial intelligence in simple words."
    },
    {
        "technique": "One-Shot",
        "prompt": "Example: Python is a programming language. Now explain Java in a similar way."
    },
    {
        "technique": "Few-Shot",
        "prompt": "Example 1: Cat -> Animal. Example 2: Rose -> Plant. Example 3: Eagle -> Bird. Now classify: Dolphin."
    },
    {
        "technique": "Role Prompting",
        "prompt": "Act as a computer science teacher and explain machine learning to a beginner."
    },
    {
        "technique": "Chain-of-Thought",
        "prompt": "Solve this problem step by step: If a student studies 2 hours per day for 5 days, how many hours do they study?"
    }
]

# 2. Sample responses
responses = []

for item in prompts:

    technique = item["technique"]
    prompt = item["prompt"]

    if technique == "Zero-Shot":
        response = "Artificial intelligence is technology that allows computers to perform tasks that normally require human intelligence."

    elif technique == "One-Shot":
        response = "Java is a programming language used to build applications and software."

    elif technique == "Few-Shot":
        response = "Dolphin -> Animal"

    elif technique == "Role Prompting":
        response = "Machine learning is a part of artificial intelligence where computers learn patterns from data."

    else:
        response = "2 hours × 5 days = 10 hours."

    responses.append({
        "technique": technique,
        "prompt": prompt,
        "generated_response": response
    })

# 3. Save prompts
prompt_df = pd.DataFrame(prompts)
prompt_df.to_csv("dataset/prompts.csv", index=False)

# 4. Save generated responses
response_df = pd.DataFrame(responses)
response_df.to_csv("dataset/generated_responses.csv", index=False)

# 5. Prompt comparison
comparison = []

for item in responses:
    comparison.append({
        "technique": item["technique"],
        "prompt_length": len(item["prompt"]),
        "response_length": len(item["generated_response"])
    })

pd.DataFrame(comparison).to_csv(
    "dataset/prompt_comparison.csv",
    index=False
)

# 6. Evaluation
evaluation = []

for item in responses:
    evaluation.append({
        "technique": item["technique"],
        "clarity": 5,
        "relevance": 5,
        "specificity": 4
    })

pd.DataFrame(evaluation).to_csv(
    "dataset/evaluation_results.csv",
    index=False
)

# 7. Prompt templates
templates = [
    {
        "technique": "Zero-Shot",
        "template": "Explain [TOPIC] in simple words."
    },
    {
        "technique": "One-Shot",
        "template": "Example: [EXAMPLE]. Now explain [TOPIC] in a similar way."
    },
    {
        "technique": "Few-Shot",
        "template": "Example 1: [A]. Example 2: [B]. Now classify [INPUT]."
    },
    {
        "technique": "Role Prompting",
        "template": "Act as a [ROLE] and explain [TOPIC]."
    },
    {
        "technique": "Chain-of-Thought",
        "template": "Solve the following problem step by step: [PROBLEM]."
    }
]

pd.DataFrame(templates).to_csv(
    "dataset/prompt_templates.csv",
    index=False
)

print("Project completed successfully!")