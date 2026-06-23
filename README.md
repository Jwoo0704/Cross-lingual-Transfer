# Cross-Lingual-Transfer

## 1. prompts_files

### 1.1. src
- test.py: main code
- We use google/gemma-E2B-it model

### 1.2. prompt_files
- csicl.txt: Code-Switching In-Context Learning (CSICL) prompt (Yoo et al., 2025)
- zero_shot_cot.txt: Chain-Of-Thought prompt without shots
- five_shot.txt: five shot prompt (with Korean shots)
- zero_shot.txt: zero shot prompt
** You can ignore "five_shot_summary_test" **


## 2. Experiment Description

### 2.1. Terminology
- **Source Language**: The LLM's actual thinking process language (fixed to English).
- **Target Language**: The language used in the demonstration shots (excluding the main instruction prompt).
- **Unseen Language**: A language that is neither the Source nor the Target Language.

### 2.2. Objective
- To observe the **Cross-lingual Transfer** phenomenon, where enhancing the model's performance in a specific Target Language leads to a relative performance improvement in Unseen Languages.
- **Hypothesis**: Among the Unseen Languages, the performance improvement will be significantly greater in languages that share similar linguistic structures with the Target Language.

### 2.3. Methodology
1. **Prompt Preparation**: Prepare various prompts (e.g., Zero-shot, Few-shot, CSICL) to evaluate and observe Multilingual AI performance.
2. **Model Loading**: Load the Hugging Face model via the `transformers` library.
3. **Evaluation**: Iteratively change the Target Language and verify in each case whether the performance increases more prominently in Unseen Languages that have linguistic similarities to the given Target Language.
