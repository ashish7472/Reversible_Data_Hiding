from flask import Flask, request, render_template, send_file
import numpy as np
import cv2
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def s_order_scan(img):
    """ Converts the image into a pixel vector using S-Order Scan """
    M, N = img.shape
    vector = []
    for i in range(M):
        if i % 2 == 0:
            vector.extend(img[i, :])
        else:
            vector.extend(img[i, ::-1])
    return np.array(vector, dtype=np.uint8)

def s_order_reshape(vector, M, N):
    """ Converts the pixel vector back into an image using S-Order Reshape """
    img = np.zeros((M, N), dtype=np.uint8)
    k = 0
    for i in range(M):
        if i % 2 == 0:
            img[i, :] = vector[k:k + N]
        else:
            img[i, :] = vector[k:k + N][::-1]
        k += N
    return img

def get_peak_value(img_vector):
    """ Finds the peak pixel value in the histogram """
    hist, bins = np.histogram(img_vector, bins=np.arange(256))
    peak = np.argmax(hist)  # Most frequent intensity value
    return peak

def embed_data(image_path, message, output_path):
    """ Embeds a secret message in the image using histogram shifting """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    M, N = img.shape
    vector = s_order_scan(img)

    P = get_peak_value(vector)  # Find peak pixel intensity

    bits = [int(bit) for char in message for bit in format(ord(char), '08b')]
    bit_idx = 0

    watermarked_vector = vector.copy()
    for i in range(len(vector)):
        if vector[i] > P:
            watermarked_vector[i] += 1  # Shift histogram
        elif vector[i] == P and bit_idx < len(bits):
            watermarked_vector[i] += bits[bit_idx]
            bit_idx += 1

    watermarked_img = s_order_reshape(watermarked_vector, M, N)
    cv2.imwrite(output_path, watermarked_img)

    return P, len(bits)

def extract_data(image_path, output_path, P, msg_length):
    """ Extracts the hidden message from the image """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    M, N = img.shape
    vector = s_order_scan(img)

    bits = []
    for i in range(len(vector)):
        if vector[i] == P and len(bits) < msg_length:
            bits.append(0)
        elif vector[i] == P + 1 and len(bits) < msg_length:
            bits.append(1)

    # Convert bits to characters
    message = ''.join(chr(int(''.join(map(str, bits[i:i+8])), 2)) 
                     for i in range(0, len(bits) - len(bits) % 8, 8))

    return message

@app.route('/', methods=['GET', 'POST'])
def index():
    embed_result = None
    extract_result = None

    if request.method == 'POST' and 'embed_image' in request.files:
        image = request.files['embed_image']
        message = request.form['message']
        image_path = os.path.join(UPLOAD_FOLDER, 'input.png')
        output_path = os.path.join(OUTPUT_FOLDER, 'embedded.png')
        image.save(image_path)
        P, msg_length = embed_data(image_path, message, output_path)
        embed_result = {'file': 'embedded.png', 'peak': P, 'msg_length': msg_length}

    if request.method == 'POST' and 'extract_image' in request.files:
        image = request.files['extract_image']
        P = int(request.form['peak'])
        msg_length = int(request.form['msg_length'])
        image_path = os.path.join(UPLOAD_FOLDER, 'embedded_input.png')
        output_path = os.path.join(OUTPUT_FOLDER, 'restored.png')
        image.save(image_path)
        message = extract_data(image_path, output_path, P, msg_length)
        extract_result = {'message': message, 'restored_image': 'restored.png'}

    return render_template('index.html', embed_result=embed_result, extract_result=extract_result)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
