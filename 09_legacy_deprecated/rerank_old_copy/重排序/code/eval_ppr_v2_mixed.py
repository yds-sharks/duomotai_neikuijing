#!/usr/bin/env python3
"""Evaluate PPR v2 vs baseline on mixed queries (text+image candidates)."""
import json, re, requests, os

os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('all_proxy', None)
os.environ.pop('ALL_PROXY', None)

RETRIEVAL_EXPORT = "/mnt/data_10/mwx/workspace/multi_modal_rag/evaluate_benchmark/rag_retrieval_export/output/20260511_151007/retrieval_export.jsonl"
PPR_V2_JSONL = "重排序/data/tmp/ppr_v2_test/ppr_v2_mixed_50.jsonl"
BASELINE_JSONL = "重排序/data/tmp/ppr_v2_test/baseline_mixed_50.jsonl"
VLLM_URL = "http://127.0.0.1:8889/v1"

# Load data
print('[INFO] Loading retrieval export...')
records_by_index = {}
with open(RETRIEVAL_EXPORT) as f:
    for line in f:
        r = json.loads(line)
        records_by_index[r['sample']['index']] = r

ppr_v2 = [json.loads(l) for l in open(PPR_V2_JSONL)]
baseline = [json.loads(l) for l in open(BASELINE_JSONL)]

# Get model
resp = requests.get(f'{VLLM_URL}/models')
model_name = resp.json()['data'][0]['id']
print(f'[INFO] Model: {model_name}')

def build_prompt(question, options, retrieval_items, topk=3):
    parts = []
    for i, item in enumerate(retrieval_items[:topk], 1):
        text = item.get('text', '').strip()
        if text:
            parts.append(f'[证据{i}] {text[:800]}')
    context = '\n'.join(parts) if parts else '无相关参考资料'
    opts = '\n'.join([f'{k}. {v}' for k, v in sorted(options.items())])
    return f'请根据以下参考资料回答选择题，只需输出选项字母（如A、B、C、D）。\n\n参考资料：\n{context}\n\n问题：{question}\n选项：\n{opts}\n\n答案是：'

def call_vllm(prompt):
    resp = requests.post(f'{VLLM_URL}/chat/completions', json={
        'model': model_name,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': 32, 'temperature': 0.0,
    }, timeout=60)
    resp.raise_for_status()
    return resp.json()['choices'][0]['message']['content'].strip()

def parse_answer(response):
    m = re.search(r'\b([A-F])\b', response.strip())
    if m: return m.group(1)
    if response and response[0] in 'ABCDEF': return response[0]
    return ''

print(f'[INFO] Evaluating {len(ppr_v2)} mixed queries...')
base_correct = ppr_correct = 0
both_correct = only_base = only_ppr = both_wrong = 0

for i, (p_result, b_result) in enumerate(zip(ppr_v2, baseline)):
    idx = p_result['index']
    record = records_by_index.get(idx)
    if not record:
        continue
    sample = record['sample']
    question = sample.get('question', '')
    options = sample.get('options', {})
    gt = sample.get('answer', '')

    b_pred = parse_answer(call_vllm(build_prompt(question, options, b_result.get('retrieval', []))))
    p_pred = parse_answer(call_vllm(build_prompt(question, options, p_result.get('retrieval', []))))
    
    b_ok = (b_pred == gt)
    p_ok = (p_pred == gt)
    if b_ok: base_correct += 1
    if p_ok: ppr_correct += 1
    if b_ok and p_ok: both_correct += 1
    elif b_ok and not p_ok: only_base += 1
    elif not b_ok and p_ok: only_ppr += 1
    else: both_wrong += 1

    if (i+1) % 10 == 0:
        print(f'  {i+1}/{len(ppr_v2)}: Base={base_correct}/{i+1} PPR_v2={ppr_correct}/{i+1}')

n = len(ppr_v2)
print(f'''
{'='*60}
混合Query端到端评测 (n={n}, beta=0.7)
{'='*60}
Baseline: {base_correct}/{n} = {100*base_correct/n:.1f}%
PPR v2:   {ppr_correct}/{n} = {100*ppr_correct/n:.1f}%
差值:     {100*(ppr_correct-base_correct)/n:+.1f}%
{'='*60}
两者都对: {both_correct}  仅Baseline对: {only_base}
仅PPR v2对: {only_ppr}  两者都错: {both_wrong}
PPR v2 净收益: {only_ppr - only_base} 题
''')

output = {'n': n, 'base_acc': base_correct/n, 'ppr_v2_acc': ppr_correct/n, 
          'diff': (ppr_correct-base_correct)/n, 'only_base': only_base, 'only_ppr': only_ppr}
with open('重排序/data/tmp/ppr_v2_test/eval_mixed_50.json', 'w') as f:
    json.dump(output, f, indent=2)
print('已保存.')
