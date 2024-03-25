CUDA_VISIBLE_DEVICES=3 python eval.py \
    --init_model_pass=best \
    --attack_method_list=natural-fgsm-pgd-cw-auto \
    --dataset=cifar10 \
    --test_batch=100 \
    --net_type=pre-res \
    --depth=18 \
    --widen_factor=10 \
    --net_module='at' \
    --save_name=cifar10-IBD-new

