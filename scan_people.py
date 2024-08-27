import argparse
import cv2
import os
import numpy as np
import shutil
import threading
from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter, run_inference

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

def ask_user_confirmation(prompt):
    while True:
        response = input(prompt + " (y/n): ").strip().lower()
        if response in ['y', 'n']:
            return response == 'y'
        print("Invalid response. Please enter 'y' or 'n'.")

def main():
    default_model_dir = 'models'
    default_model = 'mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    default_labels = 'coco_labels.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir, default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use.', default=0)
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='classifier score threshold')
    parser.add_argument('--person_number', type=int, default=1,
                        help='Identifier for the scanned person (used for directory naming)')
    args = parser.parse_args()

    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)

    # Directory setup
    person_number = args.person_number
    count_images_saved = 0
    max_images = 20  # Maximum number of images to capture

    base_dir = 'scanned_people'
    person_dir = os.path.join(base_dir, str(person_number))

    if not os.path.isdir(base_dir):
        os.mkdir(base_dir)

    if os.path.isdir(person_dir):
        if ask_user_confirmation(f"The folder '{person_dir}' already exists. Do you want to replace its contents?"):
            shutil.rmtree(person_dir)
        else:
            print("Exiting without making changes.")
            return

    os.mkdir(person_dir)
    os.mkdir(os.path.join(person_dir, 'png'))
    os.mkdir(os.path.join(person_dir, 'npy'))
    
    display_width = 854
    display_height = 480 

    cap = cv2.VideoCapture('rtsp://admin:mmuzte123@192.168.254.2:554/cam/realmonitor?channel=1&subtype=0')
    fresh = FreshestFrame(cap)

    try:
        while True:
            cnt, frame = fresh.read()
            if frame is None:
                break

            if count_images_saved >= max_images:
                break
            
            height, width, _ = frame.shape
            
            cv2_im = frame
            cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
            cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
            run_inference(interpreter, cv2_im_rgb.tobytes())
            objs = get_objects(interpreter, args.threshold)[:args.top_k]
            display_frame = cv2.resize(cv2_im.copy(), (display_width, display_height))

            cv2_im_with_labels = append_objs_to_img(display_frame, inference_size, objs, labels)

            for obj in objs:
                bbox = obj.bbox.scale(width / inference_size[0], height / inference_size[1])
                x0, y0 = int(bbox.xmin), int(bbox.ymin)
                x1, y1 = int(bbox.xmax), int(bbox.ymax)

                if obj.score > 0.8 and count_images_saved < max_images:
                    img_cut = cv2_im[y0:y1, x0:x1]
                    img_cut = cv2.resize(img_cut, (96, 96))  # Resize to 96x96

                    cv2.imwrite(f'scanned_people/{person_number}/png/img_{count_images_saved}.png', img_cut)
                    np.save(f'scanned_people/{person_number}/npy/img_{count_images_saved}', img_cut)

                    print(f"Saved image {count_images_saved} as PNG and Numpy array.")

                    count_images_saved += 1

                    if count_images_saved >= max_images:
                        print("Captured the maximum number of images.")
                        break

            cv2.imshow('frame', cv2_im_with_labels)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        fresh.release()
        cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0 + 30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()