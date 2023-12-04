for num_layers in {3,6,9}
do
for d_model in {64,128,256}
do
for lr in {1e-2,1e-3,1e-4}
do

python main.py --output_dir output \
                   --name BeijingPM25Quality_fromScratch_Regression \
                   --records_file Regression_records_train_test.xls \
                   --data_dir ./BeijingPM25Quality/ \
                   --data_class tsra \
                   --mode train_test \
                   --pattern TRAIN \
                   --test_pattern TEST\
                   --val_ratio 0.2 \
                   --epochs 200 \
                   --lr $lr \
                   --batch_size 128 \
                   --optimizer RAdam \
                   --pos_encoding learnable \
                   --task regression \
                   --shuffle True \
                   --model transformer \
                   --d_model $d_model \
                   --dim_feedforward 256\
                   --num_layers $num_layers \

done
done
done