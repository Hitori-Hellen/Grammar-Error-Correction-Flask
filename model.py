from happytransformer import HappyTextToText
from happytransformer import TTSettings

mymodel = HappyTextToText("T5", model_name="./mymodel")

beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=20)
example_2 = "grammar: I am enjoys, writtings articles ons AI and I also enjoyed write articling on AI."

result_2 = mymodel.generate_text(example_2, args=beam_settings)
print(result_2.text)