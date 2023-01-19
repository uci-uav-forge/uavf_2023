python create_coco_tf_record.py \
      --train_image_dir="../output/train" \
      --val_image_dir="../output/validation" \
      --test_image_dir="../output/test" \
      --train_annotations_file="../output/train/coco.json" \
      --val_annotations_file="../output/validation/coco.json" \
      --testdev_annotations_file="../output/test/coco.json" \
      --output_dir="../output/tfrecord"

#old protobuf version was 3.19.6
#EXTREMELY SCUFFED FIX FOR PROTOBUF ERROR:
#https://stackoverflow.com/a/72494013/14587004
# also see https://stackoverflow.com/questions/31003994/where-is-site-packages-located-in-a-conda-environment
# for figuring out where site packages is.