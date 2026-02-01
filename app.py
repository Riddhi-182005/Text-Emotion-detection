from transformers import pipeline

# Load pre-trained emotion model
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Take input from user
text = input("Enter a sentence to detect emotion: ")

# Predict emotion
predictions = emotion_model(text)[0]
best_emotion = max(predictions, key=lambda x: x["score"])

# Print results
print("\nResult:")
print("Emotion:", best_emotion['label'])
print("Confidence:", round(best_emotion['score']*100, 2), "%")
