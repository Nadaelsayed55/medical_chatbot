# Diabetes Personalized ML + Medical Assistant

This is the final version of the project:

- FastAPI backend
- Random Forest prediction
- BMI calculated from height + weight
- Human-readable age groups instead of codes
- Arabic user-friendly medical form
- Gemini explanation after prediction
- Personalized medical chatbot after the prediction
- Chatbot receives the exact patient context and ML result
- Conversation history is sent with follow-up questions
- Dashboard-inspired color palette

## 1. Project structure

```text
diabetes_personalized_assistant/
│
├── main.py
├── index.html
├── style.css
├── script.js
├── requirements.txt
├── .env.example
├── .gitignore
└── Clean_diabetes_data.csv   <-- put your CSV here
```

The code also accepts the original filename:

`diabetes_012_health_indicators_BRFSS2015.csv`

## 2. Dataset

You do NOT need to change the dataset just for the new UI.

The UI converts:

- Height + Weight -> BMI
- Age group label -> original BRFSS Age code (1..13)
- Arabic Yes/No choices -> 0/1

The model continues to use the same target columns:

- `Diabetes_012`
- `Diabetes_Status`

The backend also reindexes the model input to the exact feature columns used during training.

## 3. Install Python packages

Use Python 3.9+.

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Add Gemini API key

Create a file named `.env` beside `main.py`:

```env
GEMINI_API_KEY=YOUR_REAL_KEY
GEMINI_MODEL=gemini-3.8-flash
```

Never put the key in `index.html` or `script.js`.

## 5. Start FastAPI

```powershell
uvicorn main:app --reload
```

Backend:

`http://127.0.0.1:8000`

API documentation:

`http://127.0.0.1:8000/docs`

Health check:

`http://127.0.0.1:8000/api/health`

## 6. Start the frontend

Open a second terminal in the same folder:

```powershell
python -m http.server 5500
```

Then open:

`http://127.0.0.1:5500/index.html`

Do not open the HTML by double-clicking the file; use the local HTTP server.

## 7. How the personalized chatbot works

After the patient clicks Analyze:

1. Frontend sends the patient's simple form data.
2. Backend calculates BMI again.
3. Backend converts the human-readable age group to the original dataset code.
4. Random Forest predicts the class.
5. Backend creates a `patient_context` object containing the patient's entered data, BMI, age label, and ML result.
6. Gemini generates the initial explanation.
7. Frontend saves `patient_context`.
8. Every chat message sends:
   - the patient's current context
   - the new question
   - recent conversation history
9. Gemini is instructed to use only the current patient's context for personalized statements and never invent patient data.

So if the patient asks:

"Why did I get this result?"

the assistant can answer using that patient's BMI, age group, blood pressure, cholesterol, activity, etc.

## 8. Important medical limitation

This is a prediction/educational assistant, not a diagnostic system.

The chatbot is instructed not to:
- claim a definitive diagnosis from the ML output
- invent patient information
- prescribe medications or doses

The UI also displays a medical disclaimer.

## 9. Streamlit app.py

The old `app.py` is no longer required for the final web version.

It can be kept as a backup, but do not run Streamlit and FastAPI as competing frontends.

The final architecture is:

```text
HTML + CSS + JavaScript
          |
          v
       FastAPI
       /api/predict
          |
          v
    Random Forest
          |
          v
 Patient Context + ML Result
          |
          +------> Gemini explanation
          |
          +------> /api/chat
                       |
                       v
              Personalized Chat
```
