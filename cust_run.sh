export CUDA_VISIBLE_DEVICES=1

python -m src.train_backbone --config conf/cifar100.json

python -m src.inference \
  --config conf/cifar100.json \
  --metric_type sinkhorn \
  --rebuild_support