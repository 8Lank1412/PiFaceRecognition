import argparse
import cv2
import os
import numpy as np
import re
import time
import signal
import threading

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels

def img_to_emb(interpreter, input_image):
    """Generates face embedding from image using the face embedding model."""
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details['index']

    # Resize input image to 96x96
    input_image_resized = cv2.resize(input_image, (96, 96))

    # Preprocess the input image
    input_image_normalized = np.expand_dims(input_image_resized, axis=0)
    input_image_normalized = np.asarray(input_image_normalized, dtype=np.uint8)

    # Set tensor and run inference
    interpreter.set_tensor(tensor_index, input_image_normalized)
    interpreter.invoke()

    # Get output tensor
    output_details = interpreter.get_output_details()[0]
    output_tensor_index = output_details['index']
    emb = interpreter.get_tensor(output_tensor_index)

    # Postprocess the output tensor
    scale, zero_point = output_details['quantization']
    emb = scale * (emb - zero_point)
    
    return emb

def get_person_from_embedding(people_labels, emb):
    """Identifies person from face embedding by comparing it to the database."""
    num_emb_check = 20
    path = 'scanned_people/'
    
    # Check if the scanned_people directory exists
    if not os.path.exists(path):
        print(f"Warning: '{path}' directory not found. Skipping person identification.")
        return "unknown", 0.0

    try:
        folders = os.listdir(path)
    except OSError as e:
        print(f"Error reading '{path}' directory: {e}. Skipping person identification.")
        return "unknown", 0.0

    folders = sorted(folders)
    averages = np.zeros(len(folders))
    folder_number = 0
    start = time.time()
    for folder in folders:
        average_one_person = 0
        try:
            files = os.listdir(os.path.join(path, folder, 'embeddings'))
        except OSError as e:
            print(f"Error reading embeddings for {folder}: {e}. Skipping this person.")
            continue

        files = sorted(files)
        checked = 0
        for file in files:
            try:
                emb2 = np.load(os.path.join(path, folder, 'embeddings', file))
                norm = np.sum((emb - emb2) ** 2)
                average_one_person += norm
                checked += 1
                if checked == num_emb_check:
                    break
            except (OSError, ValueError) as e:
                print(f"Error loading embedding file {file} for {folder}: {e}. Skipping this file.")
                continue

        if checked > 0:
            average_one_person /= checked
            averages[folder_number] += average_one_person
        folder_number += 1

    if folder_number == 0:
        print("No valid embeddings found. Skipping person identification.")
        return "unknown", 0.0

    who_is_on_pic = 0
    lowest_norm_found = 10
    run = 0
    end = time.time()
    for average in averages[:folder_number]:
        run += 1
        if average < 0.5 and average < lowest_norm_found:
            lowest_norm_found = average
            who_is_on_pic = run

    confidence = max(0, 1 - lowest_norm_found)
    return people_labels.get(who_is_on_pic, "Unknown"), confidence

def signal_handler(sig, frame):
    global is_running
    print('You pressed Ctrl+C! Stopping the program...')
    is_running = False

class FreshestFrame(threading.Thread):
    def __init__(self, capture, name='FreshestFrame'):
        self.capture = capture
        assert self.capture.isOpened()
        self.cond = threading.Condition()
        self.running = False
        self.frame = None
        self.latestnum = 0
        self.callback = None
        super().__init__(name=name)
        self.start()

    def start(self):
        self.running = True
        super().start()

    def release(self, timeout=None):
        self.running = False
        self.join(timeout=timeout)
        self.capture.release()

    def run(self):
        counter = 0
        while self.running:
            (rv, img) = self.capture.read()
            assert rv
            counter += 1
            with self.cond:
                self.frame = img if rv else None
                self.latestnum = counter
                self.cond.notify_all()
            if self.callback:
                self.callback(img)

    def read(self, wait=True, seqnumber=None, timeout=None):
        with self.cond:
            if wait:
                if seqnumber is None:
                    seqnumber = self.latestnum+1
                if seqnumber < 1:
                    seqnumber = 1
                rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
                if not rv:
                    return (self.latestnum, self.frame)
            return (self.latestnum, self.frame)
        
def main():
    default_model_dir = 'models'
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    default_face_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    default_emb_model = 'Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--face_model', help='face detection model path',
                        default=os.path.join(default_model_dir, default_face_model))
    parser.add_argument('--emb_model', help='face embedding model path',
                        default=os.path.join(default_model_dir, default_emb_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=5,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default=0)
    parser.add_argument('--threshold', type=float, default=0,
                        help='classifier score threshold')
    parser.add_argument('--edge_tpu', action='store_true', help='Use EdgeTPU')
    parser.add_argument('--skip_frames', type=int, default=0, help='Number of frames to skip between each processed frame')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    
    print('Loading face detection model: {}'.format(args.face_model))
    face_interpreter = make_interpreter(args.face_model)
    face_interpreter.allocate_tensors()
    
    print('Loading face embedding model: {}'.format(args.emb_model))
    if args.edge_tpu:
        interpreter_emb = make_interpreter(args.emb_model, device='edgetpu')
    else:
        interpreter_emb = make_interpreter(args.emb_model)
    interpreter_emb.allocate_tensors()
    
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    face_inference_size = input_size(face_interpreter)

    # Load person labels
    people_labels = load_labels('people_labels.txt')
    
    global is_running
    is_running = True

    signal.signal(signal.SIGINT, signal_handler)

    cap = cv2.VideoCapture('rtsp://admin:mmuzte123@192.168.254.2:554/cam/realmonitor?channel=1&subtype=0')
    fresh = FreshestFrame(cap)

    frame_count = 0
    
    display_width = 854
    display_height = 480

    try:
        while is_running:
            frame_count, frame = fresh.read(seqnumber=frame_count+1)
            if frame is None:
                break
            
            # Skip frames based on the skip_frames parameter
            if (frame_count - 1) % (args.skip_frames + 1) != 0:
                continue
            
            cv2_im = frame

            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(interpreter, args.threshold)[:args.top_k]
            cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels, frame_count, face_interpreter, face_inference_size, interpreter_emb, people_labels)

            # Resize the frame for display
            display_frame = cv2.resize(cv2_im, (display_width, display_height))

            cv2.imshow('frame', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping the program...")
    finally:
        # Release everything when the job is finished
        fresh.release()
        cv2.destroyAllWindows()
        
def append_objs_to_img(cv2_im, inference_size, objs, labels, frame_count, face_interpreter, face_inference_size, interpreter_emb, people_labels):
    identified_persons = {}  # Dictionary to store person:confidence pairs
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    
    # Create a separate copy of the frame for cropping
    crop_frame = cv2_im.copy()
    
    # First pass: identify all persons
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        label = labels.get(obj.id, obj.id)

        if label == 'person':
            bbox_width = x1 - x0
            bbox_height = y1 - y0

            # Custom crop logic
            if bbox_height < 1.7 * bbox_width or bbox_width > bbox_height:
                person_crop = crop_frame[y0:y1, x0:x1]
            else:
                crop_height = int(1 * bbox_width)
                new_y1 = min(height, y0 + crop_height)
                person_crop = crop_frame[y0:new_y1, x0:x1]

            # Check if person_crop is empty or out of bounds
            if person_crop.size == 0:
                continue

            # Optionally enlarge the person crop
            enlarged_person_crop = cv2.resize(person_crop, (person_crop.shape[1] * 2, person_crop.shape[0] * 2))

            # Perform face detection on the enlarged person image
            try:
                person_crop_rgb = cv2.cvtColor(enlarged_person_crop, cv2.COLOR_BGR2RGB)
                person_crop_rgb = cv2.resize(person_crop_rgb, face_inference_size)
            except cv2.error as e:
                print(f"Error converting person crop to RGB or resizing for frame {frame_count}, person ID {obj.id}: {e}")
                continue

            run_inference(face_interpreter, person_crop_rgb.tobytes())
            face_objs = get_objects(face_interpreter, 0.1)  # You may want to adjust this threshold

            if face_objs:
                face_obj = face_objs[0]  # Process only the first detected face
                face_bbox = face_obj.bbox.scale(enlarged_person_crop.shape[1] / face_inference_size[0],
                                                enlarged_person_crop.shape[0] / face_inference_size[1])
                face_x0, face_y0 = int(face_bbox.xmin), int(face_bbox.ymin)
                face_x1, face_y1 = int(face_bbox.xmax), int(face_bbox.ymax)

                # Ensure square face crop
                face_size = max(face_x1 - face_x0, face_y1 - face_y0)
                face_center_x = (face_x0 + face_x1) // 2
                face_center_y = (face_y0 + face_y1) // 2
                new_face_x0 = max(0, face_center_x - face_size // 2)
                new_face_y0 = max(0, face_center_y - face_size // 2)
                new_face_x1 = min(enlarged_person_crop.shape[1], new_face_x0 + face_size)
                new_face_y1 = min(enlarged_person_crop.shape[0], new_face_y0 + face_size)

                # Crop the square face image
                face_crop = enlarged_person_crop[new_face_y0:new_face_y1, new_face_x0:new_face_x1]

                if face_crop.size > 0:
                    # Generate face embedding
                    face_img_resized = cv2.resize(face_crop, (96, 96))
                    face_emb = img_to_emb(interpreter_emb, face_img_resized)
                    
                    # Identify the person based on the embedding
                    person, confidence = get_person_from_embedding(people_labels, face_emb[0])

                    if person != "unknown":
                        if person not in identified_persons or confidence > identified_persons[person]['confidence']:
                            identified_persons[person] = {
                                'confidence': confidence,
                                'bbox': (x0, y0, x1, y1)
                            }

    # Second pass: draw bounding boxes and labels only for the highest confidence detection of each person
    for person, data in identified_persons.items():
        x0, y0, x1, y1 = data['bbox']
        confidence = data['confidence']
        
        person_label = f'{person} ({confidence:.2f})'
        bbox_color = (0, 255, 0)  # Green color for recognized faces

        # Draw person bounding box
        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), bbox_color, 5)

        # Draw label
        (text_width, text_height), baseline = cv2.getTextSize(person_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2_im = cv2.rectangle(cv2_im, (x0, y0 - text_height - 10), (x0 + text_width + 10, y0), bbox_color, -1)
        cv2_im = cv2.putText(cv2_im, person_label, (x0 + 5, y0 - 5),
                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Third pass: label remaining unidentified persons as "Unknown"
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        label = labels.get(obj.id, obj.id)

        if label == 'person':
            # Check if this bounding box overlaps significantly with any identified person
            is_identified = any(
                overlap((x0, y0, x1, y1), data['bbox']) > 0.5 for data in identified_persons.values()
            )

            if not is_identified:
                person_label = "Unknown"
                bbox_color = (0, 0, 255)  # Red color for unrecognized faces

                # Draw person bounding box
                cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), bbox_color, 5)

                # Draw label
                (text_width, text_height), baseline = cv2.getTextSize(person_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                cv2_im = cv2.rectangle(cv2_im, (x0, y0 - text_height - 10), (x0 + text_width + 10, y0), bbox_color, -1)
                cv2_im = cv2.putText(cv2_im, person_label, (x0 + 5, y0 - 5),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return cv2_im

def overlap(box1, box2):
    # Calculate the overlap ratio between two bounding boxes
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / float(area1 + area2 - intersection)

if __name__ == '__main__':
    main()
