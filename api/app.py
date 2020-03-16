import os
from os import path
import io
import zipfile
from pathlib import Path
from fastai.vision import *
from fastai.basic_train import load_learner
from fastapi import FastAPI, File, UploadFile
from pathlib import Path
import shutil

app = FastAPI()


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

def predict_many_zipped(zip_file):
    'function takes a test dataset and returns prediction'
    
    Path("./test_dir").mkdir(parents=True, exist_ok=True)
    # return test_dir
    # # # load the learner

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall('./test_dir')

    results = []

    learn = load_learner(path='./models', file='ciat_image_model.pkl',
                         test=ImageList.from_folder('./test_dir'))
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
    
    path_dir = pathlib.Path('./test_dir', ignore_errors=True)
    shutil.rmtree(path_dir)

    return {
        'results': results
    }

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/predict_single_image')
async def predict(file: UploadFile = File(...)):
    # return {"filename": file.filename}
    return predict_single(file.file)

@app.post('/predict_batch_zipped')
async def predict_batch_zipped(file: UploadFile = File(...)):
    '''
    predict batch
    '''
    return predict_many_zipped(file.file)

# @app.get('/predict_batch')
# def predict_batch():
#     '''
#     predict batch
#     '''
#     return predict_many()

