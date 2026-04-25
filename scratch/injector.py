import re

with open('scratch/merged.py', 'r', encoding='utf-8') as f:
    merged = f.read()

# Extract KNOWN_LOCAL_MODELS
m = re.search(r'(KNOWN_LOCAL_MODELS = \[.*?\]\n)', merged, re.DOTALL)
known_local_str = m.group(1)

# Extract MODELS
m = re.search(r'(MODELS = \[.*?\]\n)', merged, re.DOTALL)
models_str = m.group(1)

# Inject into ollama_catalog.py
with open('src/neuralbrok/ollama_catalog.py', 'r', encoding='utf-8') as f:
    catalog_content = f.read()

catalog_content = re.sub(r'KNOWN_LOCAL_MODELS = \[.*?\]\n', known_local_str, catalog_content, flags=re.DOTALL)

with open('src/neuralbrok/ollama_catalog.py', 'w', encoding='utf-8') as f:
    f.write(catalog_content)

# Inject into models.py
with open('src/neuralbrok/models.py', 'r', encoding='utf-8') as f:
    models_content = f.read()

models_content = re.sub(r'MODELS = \[.*?\]\n\nFALLBACK_MODELS = MODELS', models_str + '\nFALLBACK_MODELS = MODELS', models_content, flags=re.DOTALL)

with open('src/neuralbrok/models.py', 'w', encoding='utf-8') as f:
    f.write(models_content)

print("Injected successfully!")
