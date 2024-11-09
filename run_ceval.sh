model_path=/mnt/share/xujing/nuaa_nlp/ckpt_sft/epoch5_batch_19999
output_path=/mnt/share/xujing/nuaa_nlp/output

python eval.py \
    --model_path ${model_path} \
    --cot False \
    --few_shot False \
    --with_prompt True \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --ntrain 5 \
    --do_save_csv False \
    --do_test False \
    --output_dir ${output_path} \