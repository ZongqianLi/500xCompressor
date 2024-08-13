import nltk
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
from collections import Counter

def calculate_metrics_averages(csv_file):
    data = pd.read_csv(csv_file)
    averages = data.mean()
    return averages

# check whether has downloaded NLTK
nltk.download('punkt')

# initialize ROUGE
rouge = Rouge()
cc = SmoothingFunction()

def compute_exact_match(reference, candidate):
    return 1 if reference.strip() == candidate.strip() else 0

def compute_f1_score(reference_tokens, candidate_tokens):
    common = Counter(reference_tokens) & Counter(candidate_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(candidate_tokens)
    recall = num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(reference, candidate):
    scores = {}

    # Compute Rouge scores
    rouge_scores = rouge.get_scores(candidate, reference)[0]
    for key in ['rouge-1', 'rouge-2', 'rouge-l']:
        scores[f'{key}-p'] = rouge_scores[key]['p']
        scores[f'{key}-r'] = rouge_scores[key]['r']
        scores[f'{key}-f'] = rouge_scores[key]['f']

    # Compute BLEU score
    reference_tokens = nltk.word_tokenize(reference)
    candidate_tokens = nltk.word_tokenize(candidate)
    scores['bleu'] = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=cc.method1)

    # Compute exact match
    scores['exact_match'] = compute_exact_match(reference, candidate)
    
    # Compute F1 score
    scores['f1'] = compute_f1_score(reference_tokens, candidate_tokens)

    return scores

def main(ref_file, cand_file, output_file):
    with open(ref_file, 'r', encoding='utf-8') as ref_f, open(cand_file, 'r', encoding='utf-8') as cand_f:
        references = ref_f.readlines()
        candidates = cand_f.readlines()

    results = []
    i = 1
    for reference, candidate in zip(references, candidates):
        print(i)
        i += 1
        result = compute_metrics(reference.strip(), candidate.strip())
        results.append(result)

    # Write results to CSV file
    with open(output_file, 'w', encoding='utf-8') as f:
        headers = [
            "line_number",
            "rouge-1-p", "rouge-1-r", "rouge-1-f",
            "rouge-2-p", "rouge-2-r", "rouge-2-f",
            "rouge-l-p", "rouge-l-r", "rouge-l-f",
            "bleu",
            "exact_match",  # Add exact match to headers
            "f1"  # Add F1 score to headers
        ]
        f.write(",".join(headers) + "\n")
        for idx, result in enumerate(results):
            line_values = [
                f"{idx + 1}",
                f"{result['rouge-1-p']:.4f}", f"{result['rouge-1-r']:.4f}", f"{result['rouge-1-f']:.4f}",
                f"{result['rouge-2-p']:.4f}", f"{result['rouge-2-r']:.4f}", f"{result['rouge-2-f']:.4f}",
                f"{result['rouge-l-p']:.4f}", f"{result['rouge-l-r']:.4f}", f"{result['rouge-l-f']:.4f}",
                f"{result['bleu']:.4f}",
                f"{result['exact_match']}",  # Include exact match in line values
                f"{result['f1']:.4f}"  # Include F1 score in line values
            ]
            f.write(",".join(line_values) + "\n")

if __name__ == "__main__":
    root = "<to bed filled>"
    # additional information like step
    add = ""
    ref_file = root + f'tar_results{add}'
    cand_file = root + f'gen_results{add}'
    output_file = root + f'evaluate_metrics{add}.csv'
    main(ref_file, cand_file, output_file)

    csv_file = output_file
    output_file = root + f"averages{add}.csv"
    averages = calculate_metrics_averages(csv_file)
    print("Average metrics:")
    print(averages)

    averages.to_csv(output_file, header=True)


