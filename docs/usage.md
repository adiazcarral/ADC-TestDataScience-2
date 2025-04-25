2. Create a virtual environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

pip install ipykernel
python -m ipykernel install --user --name appliances-forecast


---

### 📝 `docs/usage.md`

```markdown
# 🚀 Usage
```
## 1. Preprocess the dataset

```bash
python src/data/preprocess.py
```