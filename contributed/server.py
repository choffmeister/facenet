from http.server import BaseHTTPRequestHandler, HTTPServer
import align.detect_face
import facenet
import json
import numpy as np
import scipy.misc
import sys
import tempfile
import tensorflow as tf
import traceback

def makeHandler(graph, sess, pnet, rnet, onet):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)

                with tempfile.NamedTemporaryFile() as temp:
                    temp.write(post_data)
                    temp.seek(0)

                    result = []
                    image_size = 160 # target size of image
                    margin = 44  # margin to face
                    minsize = 20  # minimum size of face
                    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
                    factor = 0.709  # scale factor

                    image = scipy.misc.imread(temp, mode='RGB')
                    faces = extract_face_images(image, pnet, rnet, onet)
                    face_images = list(map(lambda f: f['image'], faces))

                    if len(faces) > 0:
                        tensor_input = sess.graph.get_tensor_by_name("input:0")
                        tensor_embeddings = sess.graph.get_tensor_by_name("embeddings:0")
                        tensor_phase_train = sess.graph.get_tensor_by_name("phase_train:0")
                        feed_dict = {tensor_input: face_images, tensor_phase_train: False}
                        embeddings = sess.run(tensor_embeddings, feed_dict=feed_dict)

                        for i in range(0, len(faces)):
                            result.append({
                                'confidence': faces[i]['confidence'],
                                'bounding_box': faces[i]['bounding_box'].tolist(),
                                'embedding': embeddings[i].tolist(),
                            })

                result_json = json.dumps(result).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(result_json)
            except:
                self.send_response(500)
                traceback.print_exc(file=sys.stdout)

    return Handler

def create_network_face_detection(gpu_memory_fraction):
    graph = tf.Graph()
    graph.as_default()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    sess.as_default()
    pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    facenet.load_model('/facenet/data/model-20170512-110547.pb')
    return graph, sess, pnet, rnet, onet

def extract_face_images(image, pnet, rnet, onet):
    image_size = 160 # target size of image
    margin = 44  # margin to face
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_size = np.asarray(image.shape)[0:2]
    bounding_boxes, _ = align.detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    nrof_samples = len(bounding_boxes)

    faces = []

    if nrof_samples > 0:
        for i in range(0, nrof_samples):
            if bounding_boxes[i][4] > 0.95:
                det = np.squeeze(bounding_boxes[i, 0:4])
                bb = np.zeros(4, dtype=np.int32)
                bb[0] = np.maximum(det[0] - margin / 2, 0)
                bb[1] = np.maximum(det[1] - margin / 2, 0)
                bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                cropped = image[bb[1]:bb[3], bb[0]:bb[2], :]
                aligned = scipy.misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                prewhitened = facenet.prewhiten(aligned)
                faces.append({
                    'confidence': bounding_boxes[i][4],
                    'bounding_box': bb,
                    'image': prewhitened
                })

    return faces

def run(port=8080):
    graph, sess, pnet, rnet, onet = create_network_face_detection(1)
    httpd = HTTPServer(('', port), makeHandler(graph, sess, pnet, rnet, onet))
    print('Starting httpd...')
    httpd.serve_forever()

if __name__ == '__main__':
    from sys import argv
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()