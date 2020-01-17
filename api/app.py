import os
import pprint
from pathlib import Path
from flask import Flask, request, jsonify
from fastai.basic_train import load_learner
from fastai.vision import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, support_credentials=True)


def get_raw_content(raw_directory):
    return [raw_file for raw_file in os.listdir(raw_directory) if os.path.isfile(os.path.join(raw_directory, raw_file))]


def prettify(res_out):
    result = []
    for f_name, cat, probs in res_out:
        d = {"file_name": f_name, "prediction": cat, "Confidence": probs}
        result.append(d)
    final_data = {"results": result}

    return final_data


def predict_single(img_file):
    'function to take image and return prediction'
    # load the learner
    learn = load_learner(path='./models', file='ciat_image_model.pkl')
    classes = learn.data.classes
    prediction = learn.predict(open_image(img_file))
    probs_list = prediction[2].numpy()
    return {
        'category': classes[prediction[1].item()],
        'probs': {c: round(float(probs_list[i]), 5) for (i, c) in enumerate(classes)}
    }


def predict_many():
    'function takes a test dataset and returns prediction'
    # # load the learner
    learn = load_learner(path='./models', file='ciat_image_model.pkl')
    classes = learn.data.classes

    test_data_dir = 'test_images/'
    data_files = get_raw_content(test_data_dir)
    file_name = []
    category = []
    probs = []
    results = []
    for img_file in data_files:
        prediction = learn.predict(open_image(
            os.path.join(Path(test_data_dir), img_file)))
        probs_list = prediction[2].numpy()

        file_name.append(img_file)
        category.append(classes[prediction[1].item()])
        probs.append({c: round(float(probs_list[i]), 5)
                      for (i, c) in enumerate(classes)})

    return {
        'file_name': file_name,
        'prediction': category,
        'confidence': probs
    }


def predict_many_pretty():
    'function takes a test dataset and returns prediction'
    # # load the learner
    learn = load_learner(path='./models', file='ciat_image_model.pkl')
    classes = learn.data.classes

    test_data_dir = 'test_images/'
    data_files = get_raw_content(test_data_dir)

    results = []
    for img_file in data_files:
        prediction = learn.predict(open_image(
            os.path.join(Path(test_data_dir), img_file)))
        probs_list = prediction[2].numpy()

        file_name = img_file
        category = classes[prediction[1].item()]
        probs = {c: round(float(probs_list[i]), 5)
                 for (i, c) in enumerate(classes)}
        d = {'file_name': file_name, 'prediction': category, 'confidence': probs}
        results.append(d)
    return {
        'results': results
    }


def predict_many_two():
    'function takes a test dataset and returns prediction'
    # # load the learner
    results = []
    learn = load_learner(path='./models', file='ciat_image_model.pkl',
                         test=ImageList.from_folder('test_images'))
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)
    thresh = 0.2
    classes = learn.data.classes
    category = [' '.join([classes[i] for i, p in enumerate(
        pred) if p > thresh]) for pred in preds]
    file_name = [f.name[:-4] for f in learn.data.test_ds.items]

    preds = preds.tolist()
    confidences = [{c: round((p), 3) for (c, p) in zip(
        classes, probs)} for probs in preds]

    d = {'file_name': file_name, 'prediction': category, 'confidence': confidences}
    results.append(d)

    return {
        'results': results
    }


def predict_many_two_pretty():
    'function takes a test dataset and returns prediction'
    # # load the learner
    results = []
    learn = load_learner(path='./models', file='ciat_image_model.pkl',
                         test=ImageList.from_folder('test_images'))
    preds, _ = learn.get_preds(ds_type=DatasetType.Test)

    thresh = 0.2
    classes = learn.data.classes
    category = [' '.join([classes[i]
                          for i, p in enumerate(pred) if p > thresh]) for pred in preds]
    file_name = [f.name for f in learn.data.test_ds.items]

    preds = preds.tolist()

    confidences = [{c: round((p), 3) for (c, p) in zip(
        classes, probs)} for probs in preds]
    mapped = zip(file_name, category, confidences)

    return list(mapped)


@app.route('/', methods=['POST', 'GET'])
def read_root():
    return {"Hello": "World"}

# route for single prediction
@app.route('/predict', methods=['POST'])
def predict():
    return jsonify(predict_single(request.files['image']))

# route for batch prediction
@app.route('/predict_batch', methods=['POST', 'GET'])
def predict_batch():
    '''
    predict batch
    '''
    return jsonify(predict_many())

# route for batch prediction
@app.route('/predict_batch_pretty', methods=['POST', 'GET'])
def predict_batch_pretty():
    '''
    predict batch pretty
    '''
    return jsonify(predict_many_pretty())

# route for batch prediction
@app.route('/predict_batch_two', methods=['POST', 'GET'])
def predict_batch_two():
    '''
    predict batch example two
    '''
    return jsonify(predict_many_two())

# route for batch prediction - prettified
@app.route('/predict_batch_two_pretty', methods=['POST', 'GET'])
def predict_batch_two_pretty():
    '''
    predict batch example two pretty
    '''
    return jsonify(prettify(predict_many_two_pretty()))


if __name__ == '__main__':
    app.run(debug=False, port=80, host='0.0.0.0')
