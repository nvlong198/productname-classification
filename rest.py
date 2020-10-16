from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
from prediction import Prediction

"""
[summary] Khởi tạo REST API bằng Flask

Returns
-------
[object]
    [description] trả về object dữ liệu yêu cầu
"""

app = Flask(__name__)
CORS(app)
pre = Prediction()

@app.route('/process', methods=['POST'])
def process():
    global pre
    data = request.get_json()
    input_string = data['input_string'] 
    input_split = input_string.split('\n')
    if len(input_string.strip()) == 0:
        return jsonify(label='Hello World')
    else:
        predict_label = pre.process(input_split[0])
        return jsonify(label=predict_label)

if __name__ == '__main__':
    app.debug = True    
    app.run(host = '0.0.0.0', port=6970) # host = '0.0.0.0'