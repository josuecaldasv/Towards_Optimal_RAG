import requests
import json
import re
from nltk.corpus import stopwords
import string
import pandas as pd
import openpyxl
import evaluate
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import ast
import tiktoken
import random
import glob
import os


model_st = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
stop_words_english = set(stopwords.words('english'))
exact_match_metric = evaluate.load("exact_match")
rouge_metric = evaluate.load("rouge")
bleu_metric = evaluate.load("bleu")
cols_to_use = [ 
                'Query', 
                'Correct Answer', 
                'Noise_0 Predicted Answer', 'Noise_0 Appended Context', 'EM Noise_0', 'EM - 2V Noise_0', 'Cosine Noise_0', 'Jaccard Noise_0',
                'Noise_20 Predicted Answer', 'Noise_20 Appended Context', 'EM Noise_20', 'EM - 2V Noise_20', 'Cosine Noise_20', 'Jaccard Noise_20',
                'Noise_40 Predicted Answer', 'Noise_40 Appended Context', 'EM Noise_40', 'EM - 2V Noise_40', 'Cosine Noise_40', 'Jaccard Noise_40',
                'Noise_60 Predicted Answer', 'Noise_60 Appended Context', 'EM Noise_60', 'EM - 2V Noise_60', 'Cosine Noise_60', 'Jaccard Noise_60',
                'Noise_80 Predicted Answer', 'Noise_80 Appended Context', 'EM Noise_80', 'EM - 2V Noise_80', 'Cosine Noise_80', 'Jaccard Noise_80',
                'Noise_100 Predicted Answer', 'Noise_100 Appended Context', 'EM Noise_100', 'EM - 2V Noise_100', 'Cosine Noise_100', 'Jaccard Noise_100'
]
    
def safe_eval(x):
    """
    Objective
        - Safely evaluate a string to convert it to a list or other data types if applicable.
    Input
        - x (str, list, any): A string to be evaluated or a list to return directly. Other data types are returned as is.
    Output
        - result (any): The evaluated result of the input string, or the input itself if it's not a string or can't be evaluated.
    """
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        if isinstance(x, str):
            return re.findall(r"'(.*?)'", x)
        else:
            return x 


def process_json(url):
    """
    Objective
        - Fetch JSON data from a given URL, split and process the response into individual JSON objects, 
          and ensure proper formatting of the 'answer' key.
    Input
        - url (str): The URL from which to fetch the JSON data.
    Output
        - data (list): A list of processed JSON objects from the URL, with formatting applied to the 'answer' 
                       key if necessary. Returns an empty list if there's an error.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        json_strings = re.split(r'\}\s*\n\s*\{', response.text)
        data = []
        for i, json_str in enumerate(json_strings):
            if i != 0:
                json_str = '{' + json_str
            if i != len(json_strings) - 1:
                json_str += '}'
            obj = json.loads(json_str)
            if 'answer' in obj:
                if not all(isinstance(el, list) for el in obj['answer']):
                    obj['answer'] = [obj['answer']]
            data.append(obj)
        return data
    except requests.RequestException as e:
        print("Request Error:", e)
        return []
    except json.JSONDecodeError as e:
        print("JSON Decode Error:", e)
        return []


def normalize_answer(text):
    """
    Objective
        - Normalize the input text by converting it to lowercase, removing punctuation, stop words, and extra whitespace.
    Input
        - text (str, float): The input text to be normalized. If it's a float, it will be converted to a string.
    Output
        - result (list of str): A list of words from the normalized text.
    """
    if isinstance(text, float):
        text = str(text)
    text_without_stop_words = " ".join([word for word in text.split() if word.lower() not in stop_words_english])
    text_white_space_fix = " ".join(text_without_stop_words.split())
    text_without_punctuation = "".join(ch for ch in text_white_space_fix if ch not in string.punctuation)
    text_lower = text_without_punctuation.lower()
    return text_lower.split()


def jaccard_similarity_formula(set1, set2):
    """
    Objective
        - Calculate the Jaccard similarity between two sets, which is the size of the intersection divided by the size of the union.
    Input
        - set1 (set): The first set to compare.
        - set2 (set): The second set to compare.
    Output
        - similarity (float): The Jaccard similarity index, or 0.0 if the union of the sets is empty.
    """
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0: 
        print('-------Error: Union is empty-------')
        print(f'Intersection: {intersection}, Union: {union}')
        return 0.0 
    return len(intersection) / len(union)



def calculate_jaccard(predicted, correct_answers):
    """
    Objective
        - Calculate the maximum Jaccard similarity between a predicted answer and a list of correct answers.
    Input
        - predicted (str): The predicted answer to compare.
        - correct_answers (list of str): A list of correct answers to calculate the similarity against.
    Output
        - max_similarity_index (float): The highest Jaccard similarity index found between the predicted and correct answers.
    """
    normalized_predicted = set(normalize_answer(predicted))
    max_similarity_index = 0
    for answer in correct_answers:
        normalized_answer = set(normalize_answer(answer))
        similarity_index = jaccard_similarity_formula(normalized_predicted, normalized_answer)
        if similarity_index > max_similarity_index:
            max_similarity_index = similarity_index
    return max_similarity_index


def apply_jaccard(row, pred, true):
    """
    Objective
        - Apply the Jaccard similarity calculation to a DataFrame row by comparing predicted and true values.
    Input
        - row (pd.Series): A row of the DataFrame containing predicted and true values.
        - pred (str): The column name for the predicted value in the DataFrame.
        - true (str): The column name for the true value in the DataFrame.
    Output
        - similarity (float): The Jaccard similarity index for the row's predicted and true values.
    """
    return calculate_jaccard(row[pred], row[true])


def calculate_exact_match(predicted, correct_answers):
    """
    Objective
        - Calculate the exact match between a predicted answer and a list of correct answers, considering the Hugging Face metric.
    Input
        - predicted (str): The predicted answer to compare.
        - correct_answers (list of str): A list of correct answers to compare the prediction against.
    Output
        - max_start_match (float): The highest exact match score found between the predicted and correct answers.
    """
    max_start_match = 0
    predicted_strip = predicted.strip()
    for answer in correct_answers:
        result = exact_match_metric.compute(references=[answer], predictions=[predicted_strip], ignore_case=True, ignore_punctuation=True)
        start_match = result["exact_match"]
        if start_match > max_start_match:
            max_start_match = start_match
    return max_start_match

def apply_exact_match(row, pred, true):
    """
    Objective
        - Apply the Hugging Face exact match calculation to a DataFrame row by comparing predicted and true values.
    Input
        - row (pd.Series): A row of the DataFrame containing predicted and true values.
        - pred (str): The column name for the predicted value in the DataFrame.
        - true (str): The column name for the true value in the DataFrame.

    Output
        - similarity (float): The exact match score for the row's predicted and true values.
    """
    return calculate_exact_match(row[pred], row[true])


def calculate_exact_match_2v(predicted, correct_answers):
    """
    Objective
        - Evaluate if the predicted answer exactly matches any of the correct answers after normalization.
          It evaluates if all the words in the correct answer are present in the predicted answer.
          This is the exact match caculation we use in the paper. 
    
    Input
        - predicted (str): The predicted answer to compare.
        - correct_answers (list of str): A list of correct answers to compare the prediction against.

    Output
        - match (int): Returns 1 if there is an exact match, otherwise returns 0.
    """
    normalized_predicted = set(normalize_answer(predicted))
    match = 0
    for answer in correct_answers:
        normalized_answer = set(normalize_answer(answer))
        if all(word in normalized_predicted for word in normalized_answer):
            match = 1
            break
    return match

def apply_exact_match_2v(row, pred, true):
    """
    Objective
        - Apply the exact match calculation using the calculate_exact_match_2v function to a DataFrame row.
    Input
        - row (pd.Series): A row of the DataFrame containing predicted and true values.
        - pred (str): The column name for the predicted value in the DataFrame.
        - true (str): The column name for the true value in the DataFrame.

    Output
        - match (int): Returns 1 if there is an exact match, otherwise returns 0.
    """
    return calculate_exact_match_2v(row[pred], row[true])

def calculate_cosine(predicted, correct_answers):
    """
    Objective
        - Calculate the cosine similarity between the predicted answer and a list of correct answers using their embeddings.
    Input
        - predicted (str): The predicted answer to compare.
        - correct_answers (list of str): A list of correct answers to calculate the similarity against.
    Output
        - cosine_similarity (float): The cosine similarity score between the predicted and correct answer embeddings.
    """
    embeddings_pred = model_st.encode(predicted, convert_to_tensor=True)
    for answer in correct_answers:
        embeddings_true = model_st.encode(answer, convert_to_tensor=True)
        cosine_similarity = util.pytorch_cos_sim(embeddings_pred, embeddings_true)
        return cosine_similarity.item()

def apply_cosine(row, pred, true):
    """
    Objective
        - Apply the cosine similarity calculation to a DataFrame row by comparing predicted and true values.
    Input
        - row (pd.Series): A row of the DataFrame containing predicted and true values.
        - pred (str): The column name for the predicted value in the DataFrame.
        - true (str): The column name for the true value in the DataFrame.
    Output
        - cosine_similarity (float): The cosine similarity score for the row's predicted and true values.
    """
    return calculate_cosine(row[pred], row[true])

def f1_score(prediction, ground_truth):
    """
    Objective
        - Calculate the F1 score between the predicted and ground truth answers, based on precision and recall.
    Input
        - prediction (str): The predicted answer.
        - ground_truth (str): The correct answer to compare the prediction against.
    Output
        - f1 (float): The F1 score, a measure of the balance between precision and recall. Returns 0.0 if there are no common tokens.
    """
    prediction_tokens = normalize_answer(prediction)
    ground_truth_tokens = normalize_answer(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def max_f1_score(prediction, ground_truths):
    """
    Objective
        - Calculate the maximum F1 score between a predicted answer and a list of ground truth answers.
    Input
        - prediction (str): The predicted answer.
        - ground_truths (list of str): A list of correct answers to calculate the F1 score against.
    Output
        - max_f1 (float): The highest F1 score found between the predicted and any of the ground truth answers.
    """
    max_f1 = 0.0
    for gt in ground_truths:
        f1 = f1_score(prediction, gt)
        if f1 > max_f1:
            max_f1 = f1
    return max_f1

def rouge(prediction, correct_answers):
    """
    Objective
        - Calculate the ROUGE-l score between a predicted answer and the correct answers.
    Input
        - prediction (str): The predicted answer.
        - correct_answers (list of str): A list of correct answers to calculate the ROUGE-1 score against.
    Output
        - rougel (float): The ROUGE-l score between the predicted answer and the correct answers.
    """
    results = rouge_metric.compute(references=[correct_answers], predictions=[prediction])
    return results["rougeL"]

def bleu(prediction, correct_answers):
    """
    Objective
        - Calculate the BLEU score between a predicted answer and the correct answers.
    Input
        - prediction (str): The predicted answer.
        - correct_answers (list of str): A list of correct answers to calculate the BLEU score against.
    Output
        - bleu (float): The BLEU score between the predicted answer and the correct answers.
    """
    results = bleu_metric.compute(references=[correct_answers], predictions=[prediction])
    return results["bleu"]


def process_context(context, qa_pipeline, query, separator):
    """
    Objective
        - Concatenate context documents, pass them through a QA pipeline with a query, and retrieve the relevant context interval and document.
    Input
        - context (list of str): A list of context documents to be concatenated and processed.
        - qa_pipeline (function): The QA pipeline function to process the query and context.
        - query (str): The query to retrieve the answer from the context.
        - separator (str): The separator used to concatenate the context documents.
    Output
        - results_dict (dict): A dictionary containing the predicted answer, concatenated context, context interval, 
          document index, and the specific document where the answer was found.
    """
    context_concat = separator.join(context)
    results = qa_pipeline(question=query, context=context_concat)
    start_idx = results['start']
    end_idx = results['end']
    interval = f"{start_idx} - {end_idx}"
    document_indexes = []
    current_index = 0
    for element in context:
        start = current_index
        end = current_index + len(element)
        document_indexes.append((start, end))
        current_index = end + len(separator)
    document_index = next((i for i, (start, end) in enumerate(document_indexes) if start <= start_idx < end), len(context) - 1)
    document = context[document_index]
    return {
        'Predicted Answer': str(results['answer']).strip(),
        'Appended Context': str(context_concat),
        'Context Interval': str(interval),
        'Document Index': str(document_index),
        'Document': str(document)
    }

def wrap_text_and_add(label, context_results):
    """
    Args:
        label: The label to identify the block of text.
        context_results: A dictionary containing the results for different noise levels.
    """
    yield f"- {label}:"
    for noise_level in ['Noise_0', 'Noise_25', 'Noise_50', 'Noise_75', 'Noise_100']:
        result = context_results.get(noise_level, {})
        answer = result.get('Predicted Answer', 'N/A')
        text = result.get('Appended Context', '')
        match = result.get('EM', False)
        jaccard = result.get('Jaccard', 0.0)
        cosine = result.get('Cosine', 0.0)
        formatted_text = " ".join(text.split())
        yield f"  - {noise_level}:"
        yield f"    - Answer              : {answer}"
        yield "    - Threshold           : 0.8"
        yield "    - Source              :"
        start = 0
        max_width = 100
        while start < len(formatted_text):
            end = min(start + max_width, len(formatted_text))
            if end < len(formatted_text):
                end = formatted_text.rfind(' ', start, end)
            if end == -1 or end <= start:
                end = start + max_width
            yield f"                          {formatted_text[start:end]}"
            start = end + 1
        yield f"    - Jaccard Index V.    : {jaccard:.2f}"
        yield f"    - Cosine Similarity V.: {cosine:.2f}"
        yield f"    - Match (EM)          : {'Yes' if match else 'No'}"
        yield f"    - Match (Jaccard)     : {'Yes' if jaccard > 0.8 else 'No'}"
        yield f"    - Match (Cosine)      : {'Yes' if cosine > 0.8 else 'No'}"


def wrap_answers(label, correct_answers):
    """
    Args:
        label: The label to identify the block of answers.
        correct_answers: A list of correct answers (strings).
    """
    yield f"{label}:"
    if isinstance(correct_answers, str) and correct_answers.startswith("[") and correct_answers.endswith("]"):
        correct_answers = eval(correct_answers)
    formatted_text = ", ".join(correct_answers) if isinstance(correct_answers, list) else correct_answers
    start = 0
    max_width = 90
    while start < len(formatted_text):
        end = min(start + max_width, len(formatted_text))
        if end < len(formatted_text):
            end = formatted_text.rfind(',', start, end)
        if end == -1 or end <= start:
            end = start + max_width
        yield f"                          {formatted_text[start:end]}"
        start = end + 2


def format_results(data):
    for index, row in data.iterrows():
        yield "=" * 120
        yield f"Question {index + 1}              : {row['Query']}"
        for line in wrap_answers("Correct Answers         :  ", row['Correct Answer']):
            yield line
        yield "-" * 120
        for noise_level in ['Noise_0', 'Noise_25', 'Noise_50', 'Noise_75', 'Noise_100']:
            context_results = {
                'Predicted Answer': row[f'{noise_level} Predicted Answer'],
                'Appended Context': row[f'{noise_level} Appended Context'],
                'EM': row[f'EM {noise_level}'],
                'Jaccard': row[f'Jaccard {noise_level}'],
                'Cosine': row[f'Cosine {noise_level}'],
            }
            yield "-" * 120
            for line in wrap_text_and_add(f"Prediction ({noise_level})", context_results):
                yield line
        yield "=" * 120
        yield "\n"

# =======================================================================================================


def read_excel_in_chunks(filename, cols_to_use, chunk_size=1000):
    """
    Objective
        - Read an Excel file in chunks and return specific columns as a pandas DataFrame.
    Input
        - filename (str): The path to the Excel file.
        - cols_to_use (list of str): List of column names to be read from the file.
        - chunk_size (int, optional): The number of rows per chunk to be returned as a DataFrame (default is 1000).
    Output
        - chunk (pd.DataFrame): A DataFrame containing the specified columns from the Excel file in chunks of size `chunk_size`.
    """
    workbook = openpyxl.load_workbook(filename, read_only=True)
    sheet = workbook.active
    rows = sheet.iter_rows(min_row=1, values_only=True)
    headers = next(rows)
    col_indices = [headers.index(col) for col in cols_to_use]
    def get_row(row):
        return [row[idx] for idx in col_indices]
    data = []
    for row in rows:
        data.append(get_row(row))
        if len(data) >= chunk_size:
            yield pd.DataFrame(data, columns=cols_to_use)
            data = []
    if data:
        yield pd.DataFrame(data, columns=cols_to_use)

def read_json_in_chunks(filename, cols_to_use, chunk_size=1000):
    """
    Objective
        - Read a JSON file in chunks and return specific fields as a pandas DataFrame.
    Input
        - filename (str): The path to the JSON file.
        - cols_to_use (list of str): List of fields to be extracted from the JSON file.
        - chunk_size (int, optional): The number of rows per chunk to be returned as a DataFrame (default is 1000).
    Output
        - chunk (pd.DataFrame): A DataFrame containing the specified fields from the JSON file in chunks of size `chunk_size`.
    """
    with open(filename, 'r') as file:
        chunk = []
        for line in file:
            chunk.append(json.loads(line))
            if len(chunk) == chunk_size:
                yield pd.DataFrame(chunk, columns=cols_to_use)
                chunk = []
        if chunk:
            yield pd.DataFrame(chunk, columns=cols_to_use)


def compute_metrics(input_file, stride):
    """
    Objective
        - Compute various metrics (F1, EM, Cosine, Jaccard, RougeL, and Bleu) for predicted answers with different noise levels, and return the results as a DataFrame.
    Input
        - input_file (str): The path to the JSON file containing the dataset of predicted and correct answers.
        - stride (int): The step size to generate noise levels from 0 to 100.
    Output
        - result_df (pd.DataFrame): A DataFrame containing the average values for each metric (F1, EM, Cosine, Jaccard, RougeL, Bleu) across different noise levels.
    """
    df = pd.read_json(input_file, orient='records', lines=True)
    df['Correct Answer'] = df['Correct Answer'].apply(safe_eval)
    noise_levels = list(range(0, 101, stride))
    f1_scores = {f'Noise_{i}': [] for i in noise_levels}
    rouge_scores = {f'Noise_{i}': [] for i in noise_levels}
    bleu_scores = {f'Noise_{i}': [] for i in noise_levels}
    em_scores = {f'Noise_{i}': df[f'EM Noise_{i}'].tolist() for i in noise_levels}
    em_scores_2v = {f'Noise_{i}': df[f'EM - 2V Noise_{i}'].tolist() for i in noise_levels}
    cosine_scores = {f'Noise_{i}': df[f'Cosine Noise_{i}'].tolist() for i in noise_levels}
    jaccard_scores = {f'Noise_{i}': df[f'Jaccard Noise_{i}'].tolist() for i in noise_levels}
    for index, row in df.iterrows():
        correct_answer = row['Correct Answer']
        for i in noise_levels:
            noise_level = f'Noise_{i}'
            f1 = max_f1_score(row[f'{noise_level} Predicted Answer'], correct_answer)
            f1_scores[noise_level].append(f1)
            rouge_score = rouge(row[f'{noise_level} Predicted Answer'], correct_answer)
            rouge_scores[noise_level].append(rouge_score)
            bleu_score = bleu(row[f'{noise_level} Predicted Answer'], correct_answer)
            bleu_scores[noise_level].append(bleu_score)
    avg_f1 = {noise_level: sum(f1_scores[noise_level]) / len(f1_scores[noise_level]) if f1_scores[noise_level] else 0 for noise_level in f1_scores}
    avg_em = {noise_level: sum(em_scores[noise_level]) / len(em_scores[noise_level]) if em_scores[noise_level] else 0 for noise_level in em_scores}
    avg_em_2v = {noise_level: sum(em_scores_2v[noise_level]) / len(em_scores_2v[noise_level]) if em_scores_2v[noise_level] else 0 for noise_level in em_scores_2v}
    avg_cosine = {noise_level: sum(cosine_scores[noise_level]) / len(cosine_scores[noise_level]) if cosine_scores[noise_level] else 0 for noise_level in cosine_scores}
    avg_jaccard = {noise_level: sum(jaccard_scores[noise_level]) / len(jaccard_scores[noise_level]) if jaccard_scores[noise_level] else 0 for noise_level in jaccard_scores}
    avg_rouge = {noise_level: sum(rouge_scores[noise_level]) / len(rouge_scores[noise_level]) if rouge_scores[noise_level] else 0 for noise_level in rouge_scores}
    avg_bleu = {noise_level: sum(bleu_scores[noise_level]) / len(bleu_scores[noise_level]) if bleu_scores[noise_level] else 0 for noise_level in bleu_scores}
    result_data = {
        'Metric': [
            'F1', 
            'EM - String',
            'EM - 2V',
            'Cosine',
            'Jaccard',
            'RougeL',
            'Bleu'
        ]
    }
    for noise_level in [f'Noise_{i}' for i in noise_levels]:
        result_data[noise_level] = [
            avg_f1[noise_level], 
            avg_em[noise_level], 
            avg_em_2v[noise_level], 
            avg_cosine[noise_level], 
            avg_jaccard[noise_level], 
            avg_rouge[noise_level], 
            avg_bleu[noise_level]
        ]
    result_df = pd.DataFrame(result_data)
    return result_df

def create_mixed_context(positive_context, negative_context, noise_level, max_total_tokens, separator=" <|> "):
    """
    Objective
        - Create a mixed context by combining positive and negative context elements based on the specified noise level, while ensuring the total tokens remain within a specified limit.
    Input
        - positive_context (list of str): A list of positive context elements to include in the combined context.
        - negative_context (list of str): A list of negative context elements to mix with the positive context.
        - noise_level (float): The proportion of the total context that should be negative (between 0 and 1).
        - max_total_tokens (int): The maximum number of tokens allowed in the combined context.
        - separator (str, optional): The separator used to concatenate the context elements (default is " <|> ").
    Output
        - final_combined_context (list of str): A list of context elements (both positive and negative) shuffled and limited by the specified token count.
    """
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_elements = len(positive_context) + len(negative_context)
    num_negative = int(total_elements * noise_level)
    num_positive = total_elements - num_negative
    positive_sample = random.sample(positive_context, min(num_positive, len(positive_context)))
    negative_sample = random.sample(negative_context, min(num_negative, len(negative_context)))
    combined_context = positive_sample + negative_sample
    random.shuffle(combined_context)
    context_concat = separator.join(combined_context)
    context_tokens = enc.encode(context_concat)
    limited_tokens = context_tokens[:max_total_tokens]
    context_concat_limited = enc.decode(limited_tokens)
    final_combined_context = context_concat_limited.split(separator)
    random.shuffle(final_combined_context)
    return final_combined_context

def get_noise_levels(stride):
    """
    Objective
        - Generate a dictionary of noise thresholds, where the keys are labels in the format 'Noise_X' (with X being the noise level)
          and the values are the corresponding noise levels normalized between 0.0 and 1.0.
    Input
        - stride (int): The step value used to generate noise levels from 0 to 100 (inclusive).
    Output
        - noise_thresholds (dict): A dictionary where the keys are in the format 'Noise_X' and the values are normalized
          floats between 0.0 and 1.0.
    """
    levels = list(range(0, 101, stride)) 
    noise_thresholds = {}
    for level in levels:
        key = f"Noise_{level}"
        value = round(level / 100, 2) 
        noise_thresholds[key] = value
    return noise_thresholds


def extract_metric(model_data, metric_name, noise_thresholds):
    metric_row = model_data[model_data['Metric'] == metric_name]
    extracted_metrics = {}

    for noise_level, value in noise_thresholds.items():
        extracted_metrics[f'{noise_level}_Mean'] = metric_row[f'{noise_level}_Mean'].values[0]
        extracted_metrics[f'{noise_level}_Std'] = metric_row[f'{noise_level}_Std'].values[0]

    return extracted_metrics

def extract_metrics_from_excel(file_path, metric_name, model_mapping, noise_thresholds):
    excel_data = pd.ExcelFile(file_path)
    metrics = {}
    for sheet_name in excel_data.sheet_names:
        model_data = pd.read_excel(excel_data, sheet_name=sheet_name)
        mapped_name = model_mapping.get(sheet_name, sheet_name)
        metrics[mapped_name] = extract_metric(model_data, metric_name, noise_thresholds)
    final_df = pd.DataFrame({
        model: {
            f'{int(noise_level * 100)}%': data[f'{noise_label}_Mean']
            for noise_label, noise_level in noise_thresholds.items()
        }
        for model, data in metrics.items()
    })
    return final_df

def get_average_metrics(input_path, model_mapping):
    files = glob.glob(os.path.join(input_path, '*.json'))
    dataframes = {}
    for file in files:
        for key in model_mapping:
            if key in file:
                model_name = model_mapping[key]
                break
        df = pd.read_json(file, orient='records', lines=True)
        jaccard_cols = [col for col in df.columns if 'Jaccard' in col]
        cosine_cols = [col for col in df.columns if 'Cosine' in col]
        selected_cols = jaccard_cols + cosine_cols
        df_filtered = df[selected_cols]
        if model_name not in dataframes:
            dataframes[model_name] = [df_filtered]
        else:
            dataframes[model_name].append(df_filtered)
    average_dataframes = {}
    for model_name, df_list in dataframes.items():
        concatenated_df = pd.concat(df_list, axis=1)
        mean_df = concatenated_df.groupby(level=0, axis=1).mean()
        average_dataframes[model_name] = mean_df
    return average_dataframes