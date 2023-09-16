- install required packages: 

        pip install labelme && \
        apt-get update && \
        apt-get install -y libgl1-mesa-dev && apt-get install -y libglib2.0-0 && \
        pip install contextlib2 && \
        pip install IPython

- convert the labelme labels training set into COCO format:

        python scripts/labelme2coco.py dataset/train \
        --output dataset/train.json

- convert the labelme labels training set into COCO format:

        python scripts/labelme2coco.py dataset/val \
        --output dataset/val.json

*Note: Inside docker

- export tfrecord

        python scripts/create_coco_tf_record.py --logtostderr \
        --train_image_dir=dataset/train_img \
        --test_image_dir=dataset/val_img \
        --train_annotations_file=dataset/train.json \
        --test_annotations_file=dataset/val.json \
        --output_dir=dataset

- visualize tfrecord `python scripts/visualize_tfrecord.py dataset/train.record dataset/labelmap.pbtxt`

- train model: `CUDA_VISIBLE_DEVICES=0 python models/research/object_detection/model_main_tf2.py --pipeline_config_path=dataset/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config --model_dir=training --alsologtostderr`

- export model

        python models/research/object_detection/exporter_main_v2.py \
        --trained_checkpoint_dir training \
        --output_directory inference_graph \
        --pipeline_config_path dataset/mask_rcnn_inception_resnet_v2_1024x1024_coco17_gpu-8.config

- test unseen images `python scripts/predict_tf2.py`