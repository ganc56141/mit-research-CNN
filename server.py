from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import sys
sys.path.append("/home/ec2-user/deployedapp/")
from cancer-detection-cnn import predict_using_CNN, convert_to_jpg

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
        
        imgstring = data['data_url']
        img_path = convert_to_jpg(imgstring)
        res = predict_using_CNN(img_path)
        if res == 'positive':
            return f"There is a high chance that you have cancer. Please seek medical attention immediately"
        elif res == 'negative':
            return f"You most likely do not have cancer. Continue staying healthy!"
        else:
            return f'cannot classify. you... are a special case'
    else:
        return 'Hello! Make a post request with the patient\'s image.'

app.run(host='0.0.0.0', port=80)
