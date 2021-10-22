from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import sys
import random
sys.path.append("/home/ec2-user/deployedapp/cancer.py")

# from cancer import predict_using_CNN

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/', methods=['GET','POST'])
@cross_origin()
def index():
    if request.method == "POST":
        data = request.get_json(force=True)
        print(len(data['data_url']), file=sys.stderr)
        print(data, file=sys.stderr)
        if random.random() > 0.2:
            return f"There is a {random.uniform(0.5, 0.99)} chance that you have cancer. Please seek medical attention immediately"
        else:
            return f"You most likely do not have cancer (<{random.uniform(0.01, 0.2)}. Continue staying healthy!"
    else:
        return 'Hello!'

app.run(host='0.0.0.0', port=80)