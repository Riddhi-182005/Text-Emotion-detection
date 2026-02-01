from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=False
)

text = input("Enter a sentence to detect emotion: ")

predictions = classifier(text)[0]

print("Detected Emotion:", predictions["label"])
print("Confidence Score:", round(predictions["score"], 2))
