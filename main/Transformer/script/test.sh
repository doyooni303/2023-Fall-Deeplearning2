# if mode is test, load_model must exist
python main.py --output_dir output \
                   --log_path logging.log \
                   --mode test \
                   --load_model model_best.pth \
                   --name $name \
                   --no_timestamp \
                   --records_file Regression_test_records.xls \
                   --data_dir ./BeijingPM25Quality/ \
                   --data_class tsra \
                   --test_pattern TEST \
                   --batch_size 256 \
                   --task regression \
                   --model transformer \

done
