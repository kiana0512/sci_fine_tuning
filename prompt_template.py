def build_prompt(example):
    """
    Few-shot prompt with strong label constraint.
    """
    label_set = "MTD, BAC, PUR, GAP, RST, CLN, CTN, IMP"

    few_shot = """Example 1:
Previous: The model is trained using simulation data.
Current: The approach improves adaptability in real-world settings.
Next: It has been shown effective for different vehicle configurations.
Label: MTD

Example 2:
Previous: Previous work provides limited insight.
Current: There is a need to explore this area further.
Next: We propose a new method to fill this gap.
Label: GAP
"""

    prompt = f"""{example['instruction']}

{few_shot}
You must respond with **one label** chosen strictly from: [{label_set}].

Input:
{example['input']}
Label:"""

    return prompt
