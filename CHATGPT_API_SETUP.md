# ChatGPT API Setup Guide for Dissertation Project

## CRITICAL SECURITY NOTICE ⚠️

**YOU MUST IMMEDIATELY:**
1. Go to https://platform.openai.com/api-keys
2. **REVOKE** the API key you shared (sk-proj-uo2sQaehea54eGNs...)
3. Generate a **NEW** API key
4. The old key is now compromised and anyone can use it!

---

## Step-by-Step Setup Instructions

### Step 1: Secure Your API Key

1. Visit https://platform.openai.com/api-keys
2. Click "Revoke" on the compromised key
3. Click "+ Create new secret key"
4. Give it a name (e.g., "Dissertation Project")
5. Copy the new API key (it starts with `sk-proj-...`)
6. **SAVE IT SECURELY** - you won't see it again!

### Step 2: Configure Your Project

1. Open the `.env` file in your project directory:
   ```
   c:\Users\Leore\Downloads\disertation_2026-main\disertation_2026-main\.env
   ```

2. Replace `YOUR_NEW_API_KEY_HERE` with your NEW API key:
   ```
   OPENAI_API_KEY=sk-proj-YOUR_NEW_KEY_HERE
   ```

3. Save the file

### Step 3: Install Required Dependencies

If you haven't already installed the OpenAI package:

```powershell
cd "c:\Users\Leore\Downloads\disertation_2026-main\disertation_2026-main"
.venv\Scripts\Activate.ps1
pip install openai python-dotenv
```

### Step 4: Run Your Dissertation Experiments

#### Option 1: Run with Jupyter Notebook (Recommended)
```powershell
cd "c:\Users\Leore\Downloads\disertation_2026-main\disertation_2026-main\notebooks"
jupyter notebook dissertation_experiments.ipynb
```

#### Option 2: Run Evaluation Script
```powershell
cd "c:\Users\Leore\Downloads\disertation_2026-main\disertation_2026-main"
python run_real_experiments.py
```

#### Option 3: Test API Connection
```powershell
python -c "from src.llm_interface import GPT4Model; model = GPT4Model(); print(model.generate('Hello, test!', 'You are a helpful assistant.'))"
```

---

## Project Configuration Details

### Your Dissertation Project Uses:

**Primary Model:**
- **GPT-4** (via OpenAI API) - Main evaluation model
- Configured in: `src/llm_interface.py`
- API key loaded from: `.env` file

**Optional Models:**
- **LLaMA-3** (local or API) - Open-source comparison
- **Phi-3** (local) - Lightweight alternative

### Evaluation Features:

1. **KPI Anomaly Detection** - Analyzing 2.47M telecom samples
2. **Microservices Fault Detection** - SockShop stress test data
3. **Comprehensive EDA** - Statistical analysis & visualizations
4. **LLM Performance Metrics** - Accuracy, Precision, Recall, F1-Score

### Generated Datasets for LLM Testing:

- `datasets/processed/llm_evaluation_balanced.csv` - 100 balanced samples (50% anomalies)
- `datasets/processed/llm_evaluation_samples.json` - Full evaluation set with prompts

---

## Cost Estimates (OpenAI API)

**GPT-4 Pricing (as of Feb 2026):**
- Input: ~$30 per 1M tokens
- Output: ~$60 per 1M tokens

**Your Project Estimates:**
- 100 samples × ~500 tokens/sample = ~50,000 tokens
- Estimated cost: **$2-5** for full evaluation run
- Adjust `N_SAMPLES` in `.env` to control costs

---

## Security Best Practices

### ✅ DO:
- Keep `.env` file private (already in `.gitignore`)
- Use environment variables for API keys
- Regenerate keys if exposed
- Monitor API usage at https://platform.openai.com/usage

### ❌ DON'T:
- Never commit `.env` to Git
- Never share API keys publicly
- Never hardcode keys in source code
- Never post keys in chat/forums

---

## Troubleshooting

### "OpenAI API key not found"
- Check `.env` file exists
- Verify `OPENAI_API_KEY` is set correctly
- Make sure no spaces around the `=` sign

### "Invalid API key"
- Key might be revoked - generate a new one
- Check for typos or extra spaces
- Ensure you copied the full key

### "Rate limit exceeded"
- Reduce `REQUESTS_PER_MINUTE` in `.env`
- Increase `DELAY_BETWEEN_REQUESTS`
- Or upgrade your OpenAI plan

### "Insufficient quota"
- Add credits at https://platform.openai.com/billing
- Check usage limits on your account

---

## Running the Full Evaluation

Once your API key is configured:

```powershell
cd "c:\Users\Leore\Downloads\disertation_2026-main\disertation_2026-main"
.venv\Scripts\Activate.ps1
python run_real_experiments.py
```

This will:
1. Load the processed KPI anomaly dataset
2. Generate prompts for GPT-4
3. Send 100 samples to GPT-4 for classification
4. Calculate performance metrics
5. Save results to `results/` directory

---

## Next Steps

1. ✅ Revoke compromised API key
2. ✅ Generate new API key
3. ✅ Add new key to `.env` file
4. ✅ Test connection with simple prompt
5. ✅ Run full dissertation experiments
6. ✅ Analyze results in Jupyter notebook

---

## Support Resources

- **OpenAI API Docs:** https://platform.openai.com/docs
- **Usage Dashboard:** https://platform.openai.com/usage
- **API Keys Management:** https://platform.openai.com/api-keys
- **Billing & Credits:** https://platform.openai.com/billing

---

**Remember:** Your dissertation project is now ready to evaluate GPT-4's performance on telecom fault detection. The setup is complete - you just need to secure your API key properly!
