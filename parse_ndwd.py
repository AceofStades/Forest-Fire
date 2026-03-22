import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed. Please run: uv pip install tensorflow")
    exit(1)

file_path = "Model/dataset/ndws/next_day_wildfire_spread_train_00.tfrecord"

raw_dataset = tf.data.TFRecordDataset(file_path)

for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    
    print("Features in the dataset:")
    for key, feature in example.features.feature.items():
        kind = feature.WhichOneof('kind')
        if kind == 'float_list':
            val = feature.float_list.value
            print(f"- {key}: float_list, length {len(val)}")
        elif kind == 'int64_list':
            val = feature.int64_list.value
            print(f"- {key}: int64_list, length {len(val)}")
        elif kind == 'bytes_list':
            val = feature.bytes_list.value
            print(f"- {key}: bytes_list, length {len(val)}")
