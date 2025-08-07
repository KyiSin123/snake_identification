from flask import Flask, render_template, request, redirect
import os
from PIL import Image, ExifTags
import numpy as np
import cv2
from werkzeug.utils import secure_filename
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output_image_uri = None
            class_name = None
            confidence = None
            if(file_path):
                # results = predict(file_path)
                # Use PIL for auto-orientation
                # pil_img = Image.open(file_path.stream)
                # pil_img = auto_orient(pil_img)
                # pil_img = pil_img.convert("RGB")

                # # Convert to OpenCV format
                # image = np.array(pil_img)
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # # Resize to 640x640
                # resized = cv2.resize(image, (640, 640))

                # # Apply CLAHE
                # lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
                # l, a, b = cv2.split(lab)
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # cl = clahe.apply(l)
                # merged = cv2.merge((cl, a, b))
                # final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                model = YOLO('library/69.pt')
                results = model(file_path)  # Predict
                if results:
                    # Save prediction image
                    results[0].save(filename=os.path.join('static/predict', 'output.jpg'))
                    output_image_uri = '/static/predict/output.jpg'

                    # Get top prediction (first box)
                    if len(results[0].boxes) > 0:
                        class_id = int(results[0].boxes.cls[0])
                        confidence = float(results[0].boxes.conf[0])
                        class_name = model.names[class_id]
                    else:
                        class_name = "No object detected"
                        confidence = 0.0
                # print(result, "result")

                # // roboflow inference code/
                # if result:
                #     if 'predictions' in result[0] and len(result[0]['predictions']['predictions']) > 0:
                #         class_name = result[0]['predictions']['predictions'][0]['class']
                #         if class_name:
                #             if(class_name == "snake"):
                #                 class_name = "သံကွင်းစွပ်(Many Banded Krait)"
                #             elif(class_name == "ngan-taw-kyar"):
                #                 class_name = "ငန်းတော်ကြား/မြွေမင်းသား(Banded Krait)"
                #             elif(class_name == "water-snake"):
                #                 class_name = "ရေမြွေ(Checkered Keelback)"
                #             elif(class_name == "linn-myway"):
                #                 class_name = "လင်းမြွေ(Oriental Ratsnake)"
                #             elif(class_name == "wolfsnake"):
                #                 class_name = "မြွေဝံပုလွေ(Wolf Snake)"
                #             elif(class_name == "python"):
                #                 class_name = "စပါးကြီး/စပါးအုံ(Python)"
                #             elif(class_name == "cobra"):
                #                 class_name = "မြွေဟောက်(King Cobra)"
                #             elif(class_name == "thit-tat-ngan"):
                #                 class_name = "သစ်တက်ငန်း(Golden Tree Snake)"
                #             elif(class_name == "green-pit-viper"):
                #                 class_name = "မြွေစိမ်းမီးခြောက်(Green Pit Viper)"
                #             else:
                #                 class_name = "Unknown"
                #         base64_str = result[0]['output_image']
                #         if base64_str:
                #             output_image_uri = f"data:image/jpeg;base64,{base64_str}"
                #         confidence = result[0]['predictions']['predictions'][0]['confidence'] * 100
                    # else:
                    #     class_name = "No predictions found"
                    # class_name = result[0]['predictions']['predictions'][0]['class']
            return render_template('result.html', image_url=output_image_uri, prediction=class_name, confidence=confidence)
    return render_template('index.html', image_url=None)

@app.route('/first-aid')
def first_aid():
    return render_template('first_aid.html')

def auto_orient(pil_image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = pil_image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)

            if orientation_value == 3:
                pil_image = pil_image.rotate(180, expand=True)
            elif orientation_value == 6:
                pil_image = pil_image.rotate(270, expand=True)
            elif orientation_value == 8:
                pil_image = pil_image.rotate(90, expand=True)
    except Exception as e:
        pass  # No EXIF or error reading orientation, skip
    return pil_image

# def predict(image_path):
#     client = InferenceHTTPClient(
#         api_url="https://serverless.roboflow.com",
#         api_key="K5v0fY2VxLURBZjLq144"
#     )

#     result = client.run_workflow(
#         workspace_name="snake-demo",
#         workflow_id="detect-count-and-visualize-12",
#         images={
#             "image": image_path
#         },
#         use_cache=True # cache workflow definition for 15 minutes
#     )
#     return result

def predict(image_path):
    model = YOLO('library/69.pt')  # Load model
    class_names = model.names
    results = model(image_path)  # Predict
    return results

if __name__ == '__main__':
    app.run(debug=True)
