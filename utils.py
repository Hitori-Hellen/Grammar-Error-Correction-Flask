from transformers import AutoTokenizer, T5ForConditionalGeneration


def _load_model():
    model = T5ForConditionalGeneration.from_pretrained("./t5normal")
    print("load_complete")
    return model


def _generate_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")
    return tokenizer
