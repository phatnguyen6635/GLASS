datapath=/home/phatnguyen/Documents/data/glass/
augpath=./datasets/dtd/images
classes=('24')
flags=($(for class in "${classes[@]}"; do echo '-d '"${class}"; done))

cd ..
python main.py \
    --gpu 0 \
    --seed 0 \
    --test ckpt \
  net \
    -b efficientnet_b7 \
    -le blocks.0 \
    -le blocks.1  \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 3 \
    --meta_epochs 100 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.75 \
    --p 0.5 \
    --step 20 \
    --limit 1000 \
  dataset \
    --distribution 2 \
    --mean 0.5 \
    --std 0.1 \
    --fg 1 \
    --rand_aug 1 \
    --batch_size 4 \
    --resize 576 \
    --downsampling 2 \
    --imagesize 576 "${flags[@]}" mvtec $datapath $augpath
    