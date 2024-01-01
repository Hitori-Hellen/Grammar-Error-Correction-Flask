from flask import Flask
from flask_cors import CORS
from flask_restx import Api, Resource
import utils

api = Api()
app = Flask(__name__)
CORS(app)
api.init_app(app)

global tokenizer
tokenizer = None

global model
model = None


@api.route("/newmodel/<string:inputtext>")
class GEC(Resource):
    def get(self, inputtext):
        inputtext = "Fix grammatical errors in this sentence: " + inputtext
        input_ids = tokenizer(inputtext, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, max_length=256)
        edited_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"result": edited_text}


if __name__ == "__main__":
    print("App run")
    tokenizer = utils._generate_tokenizer()
    model = utils._load_model()
    app.run(debug=True, host="0.0.0.0", port=5000)
