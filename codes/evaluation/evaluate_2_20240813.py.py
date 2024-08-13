import nltk
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import json
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

    if not reference or not candidate:
        print(f"Empty reference or candidate detected. Reference: '{reference}', Candidate: '{candidate}'")
        raise ValueError("Reference or candidate is empty.")
    
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

    # Compute SQuAD F1 score
    scores['squad_f1'] = compute_f1_score(reference_tokens, candidate_tokens)

    return scores

def main(ref_file, cand_file, output_file):
    with open(ref_file, 'r', encoding='utf-8') as ref_f, open(cand_file, 'r', encoding='utf-8') as cand_f:
        references = [json.loads(line.strip()) for line in ref_f.readlines()]
        candidates = cand_f.readlines()

    results = []
    i = 1
    for reference, candidate in zip(references, candidates):
        print(i)
        i += 1
        candidate = candidate.strip()
        if isinstance(reference, list):
            metrics = []
            for ref in reference:
                ref = ref.strip()
                try:
                    metric = compute_metrics(ref, candidate)
                    metrics.append(metric)
                except ValueError as e:
                    print(f"Skipping computation for reference: '{ref}', candidate: '{candidate}' due to error: {e}")
            # choose the largest rouge-l-f
            if metrics:
                result = max(metrics, key=lambda x: x['rouge-l-f'])
            else:
                result = {}  # In case all references failed
        else:
            reference = reference.strip()
            try:
                result = compute_metrics(reference, candidate)
            except ValueError as e:
                print(f"Skipping computation for reference: '{reference}', candidate: '{candidate}' due to error: {e}")
                result = {}
        results.append(result)

    # Write results to CSV file
    with open(output_file, 'w', encoding='utf-8') as f:
        headers = [
            "line_number",
            "rouge-1-p", "rouge-1-r", "rouge-1-f",
            "rouge-2-p", "rouge-2-r", "rouge-2-f",
            "rouge-l-p", "rouge-l-r", "rouge-l-f",
            "bleu",
            "exact_match",
            "squad_f1"
        ]
        f.write(",".join(headers) + "\n")
        for idx, result in enumerate(results):
            if result:  # Check if result is not empty
                line_values = [
                    f"{idx + 1}",
                    f"{result.get('rouge-1-p', 0):.4f}", f"{result.get('rouge-1-r', 0):.4f}", f"{result.get('rouge-1-f', 0):.4f}",
                    f"{result.get('rouge-2-p', 0):.4f}", f"{result.get('rouge-2-r', 0):.4f}", f"{result.get('rouge-2-f', 0):.4f}",
                    f"{result.get('rouge-l-p', 0):.4f}", f"{result.get('rouge-l-r', 0):.4f}", f"{result.get('rouge-l-f', 0):.4f}",
                    f"{result.get('bleu', 0):.4f}",
                    f"{result.get('exact_match', 0)}",
                    f"{result.get('squad_f1', 0):.4f}"
                ]
                f.write(",".join(line_values) + "\n")

if __name__ == "__main__":
    root = "<to be filled>"
    ref_file = "<to be filled>"
    cand_file = root + 'gen_results'
    output_file = root + 'evaluate_metrics.csv'
    main(ref_file, cand_file, output_file)

    csv_file = output_file
    output_file = root + f"averages.csv"
    averages = calculate_metrics_averages(csv_file)
    print("Average metrics:")
    print(averages)

    averages.to_csv(output_file, header=True)
