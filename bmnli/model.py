from transformers import AutoTokenizer, AutoModelForSequenceClassification


def get_classifier(model_name, num_labels=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels)
    return model, tokenizer
