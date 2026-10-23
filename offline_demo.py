"""Zero-download emergency console demo. It is a heuristic, NOT the assessed GPT-2 model."""
from app import heuristic_predict

SAMPLES = [
    ("What are common symptoms of diabetes?", "Which signs may indicate that someone has diabetes?"),
    ("Can aspirin cause stomach bleeding?", "How can stomach bleeding be prevented?"),
    ("What causes high blood pressure?", "Why might a person develop hypertension?"),
    ("Is insulin used to treat type 1 diabetes?", "What is the role of insulin in type 1 diabetes?"),
    ("Does a sore throat always mean strep infection?", "What are the symptoms of a urinary tract infection?"),
]

print("MP-C42 emergency demo - OFFLINE HEURISTIC ONLY (not GPT-2 fine-tuning).\n")
for index, (q1, q2) in enumerate(SAMPLES, start=1):
    label, score, note = heuristic_predict(q1, q2)
    print(f"Example {index}\nQ1: {q1}\nQ2: {q2}\nPrediction: {label} | score: {score:.1%}\n")
print("Enter your own pair (blank Q1 exits).")
while True:
    q1 = input("Question 1: ").strip()
    if not q1:
        break
    q2 = input("Question 2: ").strip()
    label, score, _ = heuristic_predict(q1, q2)
    print(f"Prediction: {label} | fallback score: {score:.1%}\n")
