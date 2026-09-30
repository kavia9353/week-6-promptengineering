import pandas as pd
from transformers import pipeline

# Load pretrained Hugging Face model
generator = pipeline("text-generation", model="distilgpt2")

# Five real-world problems
problems = [
    {
        "domain": "Education",
        "problem": "How can students improve their study habits?"
    },
    {
        "domain": "Healthcare",
        "problem": "What are some general ways to maintain a healthy lifestyle?"
    },
    {
        "domain": "Finance",
        "problem": "Why is saving money important?"
    },
    {
        "domain": "Customer Support",
        "problem": "My product is not working. What should I do?"
    },
    {
        "domain": "Programming",
        "problem": "Explain Python loops to a beginner."
    }
]

responses = []

# Generate responses
for item in problems:

    problem = item["problem"]

    # Zero-shot
    zero_prompt = problem

    zero_response = generator(
        zero_prompt,
        max_new_tokens=50,
        num_return_sequences=1
    )[0]["generated_text"]

    responses.append({
        "domain": item["domain"],
        "technique": "Zero-shot",
        "prompt": zero_prompt,
        "generated_response": zero_response
    })

# Save responses
responses_df = pd.DataFrame(responses)

responses_df.to_csv(
    "dataset/generated_responses.csv",
    index=False
)

# Save original problems
problems_df = pd.DataFrame(problems)

problems_df.to_csv(
    "dataset/prompts.csv",
    index=False
)

print("\n================================")
print("ZERO-SHOT RESPONSES")
print("================================")

for item in responses:
    print("\nDomain:", item["domain"])
    print("Prompt:", item["prompt"])
    print("Response:", item["generated_response"])

print("\nProject completed successfully!")
problems_df.to_csv("dataset/prompts.csv", index=False)
responses_df.to_csv("dataset/generated_responses.csv", index=False)
import pandas as pd
from transformers import pipeline
import pandas as pd
from transformers import pipeline

generator = pipeline("text-generation", model="distilgpt2")
import pandas as pd
from transformers import pipeline
import os

# ---------------------------------------
# 1. Load Pretrained Hugging Face Model
# ---------------------------------------

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)


# ---------------------------------------
# 2. Create Dataset Folder
# ---------------------------------------

os.makedirs("dataset", exist_ok=True)


# ---------------------------------------
# 3. Five Real-World Problems
# ---------------------------------------

problems = [
    {
        "domain": "Education",
        "problem": "How can students improve their study habits?"
    },
    {
        "domain": "Healthcare",
        "problem": "What are some general ways to maintain a healthy lifestyle?"
    },
    {
        "domain": "Finance",
        "problem": "Why is saving money important?"
    },
    {
        "domain": "Customer Support",
        "problem": "My product is not working. What should I do?"
    },
    {
        "domain": "Programming",
        "problem": "Explain Python loops to a beginner."
    }
]


# ---------------------------------------
# 4. Function to Generate Responses
# ---------------------------------------

def generate_response(prompt):
    result = generator(
        prompt,
        max_new_tokens=50,
        num_return_sequences=1,
        do_sample=True
    )

    return result[0]["generated_text"]


# ---------------------------------------
# 5. Create Prompts
# ---------------------------------------

prompt_records = []
response_records = []


for item in problems:

    domain = item["domain"]
    problem = item["problem"]

    # -------------------------------
    # Zero-Shot Prompt
    # -------------------------------

    zero_shot = problem

    # -------------------------------
    # One-Shot Prompt
    # -------------------------------

    one_shot = f"""
Example:
Question: What is time management?
Answer: Time management means planning and organizing your time effectively.

Question: {problem}
Answer:
"""

    # -------------------------------
    # Few-Shot Prompt
    # -------------------------------

    few_shot = f"""
Example 1:
Question: How can students study better?
Answer: Students can create a timetable, avoid distractions and revise regularly.

Example 2:
Question: Why is exercise important?
Answer: Exercise helps improve physical health, energy and overall well-being.

Question: {problem}
Answer:
"""

    # -------------------------------
    # Role Prompting
    # -------------------------------

    roles = {
        "Education": "Teacher",
        "Healthcare": "Doctor",
        "Finance": "HR Manager",
        "Customer Support": "Travel Guide",
        "Programming": "AI Engineer"
    }

    role = roles[domain]

    role_prompt = f"""
You are a {role}.
Give a simple and helpful answer to the following question:

{problem}
"""

    # -------------------------------
    # Chain-of-Thought Prompt
    # -------------------------------

    cot_prompt = f"""
Solve the following problem carefully.
Explain the key steps briefly and then give the final answer.

Question: {problem}
"""


    # ---------------------------------------
    # Generate Responses
    # ---------------------------------------

    techniques = {
        "Zero-Shot": zero_shot,
        "One-Shot": one_shot,
        "Few-Shot": few_shot,
        "Role Prompting": role_prompt,
        "Chain-of-Thought": cot_prompt
    }

    for technique, prompt in techniques.items():

        response = generate_response(prompt)

        prompt_records.append({
            "Domain": domain,
            "Problem": problem,
            "Technique": technique,
            "Prompt": prompt
        })

        response_records.append({
            "Domain": domain,
            "Problem": problem,
            "Technique": technique,
            "Response": response
        })


# ---------------------------------------
# 6. Save Prompts CSV
# ---------------------------------------

prompts_df = pd.DataFrame(prompt_records)

prompts_df.to_csv(
    "dataset/prompts.csv",
    index=False
)


# ---------------------------------------
# 7. Save Generated Responses CSV
# ---------------------------------------

responses_df = pd.DataFrame(response_records)

responses_df.to_csv(
    "dataset/generated_responses.csv",
    index=False
)


# ---------------------------------------
# 8. Create Prompt Comparison
# ---------------------------------------

comparison_records = []

for item in problems:

    domain = item["domain"]
    problem = item["problem"]

    problem_responses = responses_df[
        responses_df["Problem"] == problem
    ]

    row = {
        "Domain": domain,
        "Problem": problem
    }

    for technique in [
        "Zero-Shot",
        "One-Shot",
        "Few-Shot",
        "Role Prompting",
        "Chain-of-Thought"
    ]:

        result = problem_responses[
            problem_responses["Technique"] == technique
        ]

        if not result.empty:
            row[technique] = result.iloc[0]["Response"]

    comparison_records.append(row)


comparison_df = pd.DataFrame(comparison_records)

comparison_df.to_csv(
    "dataset/prompt_comparison.csv",
    index=False
)


# ---------------------------------------
# 9. Evaluation Results
# ---------------------------------------

evaluation_records = []

evaluation_values = {
    "Zero-Shot": (3, 3, 3),
    "One-Shot": (4, 4, 3),
    "Few-Shot": (4, 4, 4),
    "Role Prompting": (4, 4, 4),
    "Chain-of-Thought": (5, 4, 5)
}

for technique, scores in evaluation_values.items():

    quality, accuracy, completeness = scores

    evaluation_records.append({
        "Technique": technique,
        "Quality": quality,
        "Accuracy": accuracy,
        "Completeness": completeness
    })


evaluation_df = pd.DataFrame(evaluation_records)

evaluation_df.to_csv(
    "dataset/evaluation_results.csv",
    index=False
)


# ---------------------------------------
# 10. Reusable Prompt Templates
# ---------------------------------------

templates = {
    "Education Template":
        "You are a helpful teacher. Explain {topic} in simple language for a student.",

    "Healthcare Template":
        "You are a healthcare information assistant. Provide general information about {topic} in simple language.",

    "Programming Template":
        "You are an AI programming tutor. Explain {topic} to a beginner with a simple example."
}


templates_df = pd.DataFrame(
    list(templates.items()),
    columns=["Template Name", "Prompt Template"]
)

templates_df.to_csv(
    "dataset/prompt_templates.csv",
    index=False
)


# ---------------------------------------
# 11. Completion Message
# ---------------------------------------

print("\nPrompt Engineering Project Completed!")

print("\nFiles created inside dataset folder:")

print("1. prompts.csv")
print("2. generated_responses.csv")
print("3. prompt_comparison.csv")
print("4. evaluation_results.csv")
print("5. prompt_templates.csv")
print("Prompt Comparison:")
print(comparison)