python -m bench.train_reference --arch resnet18  --dataset cifar100 --epochs 200 --out results/reference/resnet18_cifar100.pt
python -m bench.train_reference --arch vgg16_bn   --dataset cifar10  --epochs 200 --out results/reference/vgg16bn_cifar10.pt
