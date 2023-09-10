import os
import cv2
import numpy as np
import tensorflow as tf
import time 
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

IMGS_PATH = "test" #  PATH TO IMAGES
MODEL_PATH = "inference_graph_count_phone_faster_95024/frozen_inference_graph.pb" # PATH TO MODEL FILE frozen_inference_graph.pb
LABEL_PATH = "count_phone_label_map.pbtxt" # PATH TO LABEL MAP FILE labelmap.pbtxt
JSON_PATH = "output_countphone/result.json"
THRESH = 0.65
DST = "output_countphone"

# Create destination folder
if not os.path.exists(DST):
    os.mkdir(DST)

def dim_sort(h,w):
  if h > w:
    return h,w
  else:
    return w,h

def load_label_map(path_label):
  map_label = {}
  map_label_v2 = []
  num_label = 0
  with open(path_label) as fp:
    line = fp.readline().strip()
    line = line.strip()
    while line:
        if (line.strip() != '' and ('item' in line.strip())):
          map_info = {}
          total = 0
          name_label = ""
          for i in range(5):
            line = fp.readline()
            data = line.strip()

            if (data != '' and ('id:' in data)):
              data = data.replace("id:","")
              data = data.replace(" ","")
              data = data.replace("'","")
              map_info["id"] = int(data)
              total = total + 1
            if (data != '' and ('name:' in data)):
              data = data.replace("name:","")
              data = data.replace(" ","")
              data = data.replace("'","")
              map_info["name"] = data
              name_label = data
              total = total + 1
            
            if(total == 2):
              break
          num_label = num_label +1
          map_label[num_label] = map_info
          map_label_v2.append(map_info)
        line = fp.readline()
  return map_label, map_label_v2, num_label

def visualize_boxes_and_labels_on_image_array(image,
    boxes,
    classes,
    scores,
    category_index,
    num_class,
    min_score_thresh):
    
  print("Current thresh",min_score_thresh)
  jsonposition = {}
  for dataLabel  in categories:
    class_name = dataLabel['name']
    jsonposition[class_name] = []

  for i in range(boxes.shape[0]):
    if scores is None or scores[i] > min_score_thresh:
      box = tuple(boxes[i].tolist())
      class_name = category_index[classes[i]]['name']
      ymin, xmin, ymax, xmax = box
      im_height, im_width = image.shape[:2]
      (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
      print('class_name: {}, left: {}, top: {}, right: {}, bottom: {}, scores: {}'.format(class_name, left, top, right, bottom, scores[i]))
      jsontmp2 = {}
      jsontmp2['x1'] = int(left)
      jsontmp2['y1'] = int(top)
      jsontmp2['x2'] = int(right)
      jsontmp2['y2'] = int(bottom)
      jsontmp2['scores'] = float(scores[i])
      jsonposition[class_name].append(jsontmp2)

  return jsonposition 

def drawBox(frame, json_data):
  
  color = (209, 209, 50)
  key_list = list(json_data.keys())
    
  for key in key_list:
      # get list of all key items
      items = json_data[key]
      for item in items:
          x1 = item['x1']
          x2 = item['x2']
          y1 = item['y1']
          y2 = item['y2']
          conf = item['scores']
      
          # Draw a bounding box.
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))

          label = '%.2f' % conf
          label = key + ": " + label

          labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
          #cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
          cv2.putText(frame, label, (x1, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 0.2,color)

for i,IMG in enumerate(os.listdir(IMGS_PATH)):
    
    # start counting time
    start_time = time.time()

    print("Processing image {} of {}".format(i+1,len(os.listdir(IMGS_PATH))))
    IMG_PATH = os.path.join(IMGS_PATH, IMG)
    OUT_PATH = os.path.join(DST, IMG) 

    # Load image
    image = cv2.imread(IMG_PATH)
    H,W,_ = image.shape

    category_index, categories, n_class = load_label_map(LABEL_PATH)

    # Load a (frozen) Tensorflow model into memory.
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        config_tf = tf.ConfigProto()
        config_tf.gpu_options.allow_growth = True
        sess = tf.Session(config=config_tf, graph=detection_graph)

    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    jsondata = {}
    line = ""

    # read image
    image = cv2.imread(IMG_PATH)

    img_arr = np.expand_dims(image, axis=0)
    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: img_arr})

    jsontmp = {}
    jsontmp = visualize_boxes_and_labels_on_image_array( 
        image,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        n_class,
        THRESH)

    jsondata[line.strip()] = jsontmp

    with open(JSON_PATH, 'w') as outfile:
        json.dump(jsondata, outfile)
        
    drawBox(image,jsontmp)
    
    cv2.imwrite(OUT_PATH, image)

    print("time detect: {}".format(time.time() - start_time))
