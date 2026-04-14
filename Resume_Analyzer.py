import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')

training_data = [
    "Python machine learning data analysis pandas numpy",
    "HTML CSS JavaScript React frontend web development",
    "Java Spring Boot backend API development",
    "Python deep learning tensorflow keras AI",
    "C++ data structures algorithms problem solving"
]

job_labels = [
    "Data Scientist",
    "Web Developer",
    "Backend Developer",
    "AI Engineer",
    "Software Engineer"
]


skill_db = {
    "Data Scientist": ["python", "machine learning", "pandas", "numpy", "statistics"],
    "Web Developer": ["html", "css", "javascript", "react", "node"],
    "Backend Developer": ["java", "spring", "api", "database"],
    "AI Engineer": ["python", "deep learning", "tensorflow", "keras"],
    "Software Engineer": ["c++", "algorithms", "data structures"]
}


def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    return text

cleaned = [preprocess(r) for r in training_data]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cleaned)

model = LogisticRegression()
model.fit(X, job_labels)

def analyze_resume():
    print("\nPaste your resume text:\n")
    user_text = input()

    clean_text = preprocess(user_text)
    vector = vectorizer.transform([clean_text])

    predicted_role = model.predict(vector)[0]

    print("\n🔍 Predicted Role:", predicted_role)

    found_skills = []
    for skill in skill_db[predicted_role]:
        if skill in clean_text:
            found_skills.append(skill)

    print("✅ Detected Skills:", found_skills)

 
    missing = [s for s in skill_db[predicted_role] if s not in found_skills]
    print("📌 Missing Skills:", missing)

def main():
    while True:
        print("\n===== Resume Analyzer =====")
        print("1. Analyze Resume")
        print("2. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            analyze_resume()
        elif choice == "2":
            print("Goodbye!")
            break
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()