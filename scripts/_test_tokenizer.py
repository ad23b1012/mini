import ast

# Syntax check
src = open('data/meld_dataset.py', encoding='utf-8').read()
ast.parse(src)
print('Syntax OK')

# Smoke test: load tokenizer offline
from data.meld_dataset import load_tokenizer
tok = load_tokenizer('microsoft/deberta-v3-base')
print(f'Tokenizer loaded OK: {tok.__class__.__name__}')
test = tok('Hello, this is a test!', return_tensors='pt')
print(f'Tokenization OK: {test["input_ids"].shape}')
