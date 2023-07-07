#absolute_path_to_dataset="/media/data/liutie/VCM/OpenImageV6-5K/"
#absolute_path_to_dataset="./dataset/val_openimage_v6"
#export DETECTRON2_DATASETS=${absolute_path_to_dataset}

for idx in 43 35; do
    # Encoding and Decoding
#    python test.py -i ${idx} -m feature_coding
    # Inference and Evaluation
    python test.py -i ${idx} -m evaluation
done