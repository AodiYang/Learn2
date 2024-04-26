from flask import Flask, request, send_from_directory, render_template, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import os
import subprocess
import pydicom
import numpy as np

import torch
from solver import Solver
import argparse
from loader import get_loader

# Get the directory of the current script (__file__ is the path to the current script).
current_script_path = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.secret_key = 'UoG2024#YAD'
UPLOAD_FOLDER = 'uploads/'
PROCESSED_FOLDER = 'processed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
# Additional configuration needed for Solver
app.config['MODEL_PATH'] = 'D:/FD/WGAN-VGG/save/WGANVGG_250000iter.ckpt'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

def process_ima_to_npy(file_path, save_path, norm_range_min=-1024.0, norm_range_max=3072.0):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ds = pydicom.dcmread(file_path)
    image_array = ds.pixel_array.astype(np.int16)
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope

    if slope != 1:
        image_array = slope * image_array.astype(np.float64)
        image_array = image_array.astype(np.int16)

    image_array += np.int16(intercept)
    normalized_image = (image_array - norm_range_min) / (norm_range_max - norm_range_min)
    npy_filename = os.path.basename(file_path).replace('.IMA', '.npy')
    np.save(os.path.join(save_path, npy_filename), normalized_image)
    return npy_filename

# Define a function to create a Namespace object from your arguments
def create_args():
    args = argparse.Namespace(
        mode='test',
        load_mode=0,
        norm_range_min=-1024.0,
        norm_range_max=3072.0,
        trunc_min=-160.0,
        trunc_max=240.0,
        # ... include all other arguments required for your Solver here ...
        device='cuda' if torch.cuda.is_available() else 'cpu',
        test_iters=250000,
        result_fig=True
    )
    return args

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.IMA'):
            filename = secure_filename(file.filename)
            ima_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(ima_file_path)

            npy_filename = process_ima_to_npy(ima_file_path, app.config['PROCESSED_FOLDER'])
            npy_file_path = os.path.join(app.config['PROCESSED_FOLDER'], npy_filename)
            return redirect(url_for('download_page', filename=npy_filename))
    return render_template('upload.html')

@app.route('/download')
def download_page():
    filename = request.args.get('filename', '')
    files = os.listdir(app.config['PROCESSED_FOLDER'])
    return render_template('download.html', files=files, filename=filename)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

@app.route('/process_npy/<filename>', methods=['GET', 'POST'])
def process_npy(filename):
    # Construct absolute paths for the input and output files
    input_file_path = os.path.join(current_script_path, app.config['PROCESSED_FOLDER'], filename)
    processed_filename = filename.replace('.npy', '_denoised.npy')
    output_file_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)

    if os.path.exists(input_file_path):
        try:
            # Prepare arguments for Solver
            args = create_args()
            args.test_patient = input_file_path
            args.saved_path = app.config['PROCESSED_FOLDER']

            # Initialize the data loader and the solver
            # You would replace the get_loader() with the actual method of loading your data
            data_loader = get_loader(args)
            solver = Solver(args, data_loader)
            solver.load_model(args.test_iters)  # Load the model weights

            # Conduct testing (denoising)
            denoised_data = solver.test()  # Ensure that the Solver.test() method returns the denoised data

            # Save the denoised data to a new file
            np.save(output_file_path, denoised_data)

            flash("File processed successfully.")
            return redirect(url_for('download_file', filename=processed_filename))
        except Exception as e:
            flash(f"Failed to process the file: {str(e)}")
            return redirect(url_for('download_page'))
    else:
        flash("File does not exist.")
        return redirect(url_for('download_page'))

if __name__ == '__main__':
    app.run(debug=True)
