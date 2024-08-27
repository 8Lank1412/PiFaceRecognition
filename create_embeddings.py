import os
import shutil
import numpy as np
import argparse
from tflite_runtime.interpreter import load_delegate
from tflite_runtime.interpreter import Interpreter

def main(scan_person, use_edgetpu):
    # Get interpreter for face embedding model
    if use_edgetpu:
        interpreter = Interpreter(
            model_path='models/Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite',
            experimental_delegates=[load_delegate('libedgetpu.so.1.0')]
        )
    else:
        interpreter = Interpreter(
            model_path='models/Mobilenet1_triplet1589223569_triplet_quant.tflite'
        )
    
    interpreter.allocate_tensors()

    path_person = f'scanned_people/{scan_person}'
    
    if not os.path.isdir(f'{path_person}/embeddings'):
        os.mkdir(f'{path_person}/embeddings')
    else:
        shutil.rmtree(f'{path_person}/embeddings')
        os.mkdir(f'{path_person}/embeddings')
    
    files = os.listdir(f'{path_person}/npy')
    files = sorted(files)

    for file in files:
        print(file)
        img = np.load(f'{path_person}/npy/{file}').reshape(1, 96, 96, 3) / 255.
        emb = img_to_emb(interpreter, img)
        np.save(f'{path_person}/embeddings/{file}', emb)

def set_input_tensor(interpreter, input):
    # Sets the input tensor.
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']
    scale, zero_point = input_details['quantization']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = np.uint8(input / scale + zero_point)

def img_to_emb(interpreter, input):
    # Returns embedding vector, using the face embedding model
    set_input_tensor(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    emb = interpreter.get_tensor(output_details['index'])
    scale, zero_point = output_details['quantization']
    emb = scale * (emb - zero_point)
    return emb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate face embeddings.')
    parser.add_argument('scan_person', type=int, help='Folder number to process.')
    parser.add_argument('--use_edgetpu', action='store_true', help='Use Edge TPU for acceleration.')
    
    args = parser.parse_args()
    main(args.scan_person, args.use_edgetpu)
