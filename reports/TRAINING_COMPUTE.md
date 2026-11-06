# Training compute by arm (forward-FLOP x steps, relative to the dense reference)

Ref = 1.00. TSR counts the k dormant candidates per group; C2 = TSR + retrain; pruning pipelines include the dense reference they start from.

| cell | replay ok | ΔF exported | TSR | TSR w/o cand. | C2 | ℓ1 fine-tune | ℓ1 scratch | hook check |
|---|---|---|---|---|---|---|---|---|
| efficientnet_b0_cifar100 | yes | -3.6% | 1.07 | 0.99 | 2.03 | -- | -- | ok |
| efficientnet_b0_cifar10 | yes | +40.2% | 1.16 | 1.07 | 2.56 | 1.31 | 1.78 | ok |
| efficientnet_b0_tiny_imagenet | yes | +4.2% | 1.26 | 1.17 | 2.30 | -- | -- | ok |
| mobilenetv2_cifar100 | yes | -11.3% | 1.03 | 0.93 | 1.92 | -- | -- | ok |
| mobilenetv2_cifar10 | yes | -11.1% | 1.03 | 0.94 | 1.92 | 1.28 | 1.69 | ok |
| mobilenetv2_tiny_imagenet | yes | -42.0% | 0.78 | 0.71 | 1.36 | -- | -- | ok |
| resnet18_cifar100 | yes | +1.8% | 1.05 | 0.98 | 2.07 | -- | -- | ok |
| resnet18_cifar10 | yes | +31.1% | 1.17 | 1.10 | 2.48 | 1.32 | 1.79 | ok |
| resnet18_imagenet100 | yes | +12.0% | 1.21 | 1.11 | 2.33 | -- | -- | ok |
| resnet18_tiny_imagenet | yes | -19.6% | 0.90 | 0.83 | 1.70 | 1.23 | 1.57 | ok |
| vgg16bn_cifar100 | yes | +27.7% | 1.10 | 1.05 | 2.37 | -- | -- | ok |
| vgg16bn_cifar10 | yes | +27.1% | 1.09 | 1.04 | 2.36 | 1.35 | 1.86 | ok |
| vgg16bn_imagenet100 | -- | -- | -- | -- | -- | -- | -- | log ends at step 10800 of 395900; archived run is not the full discovery run |
| vgg16bn_tiny_imagenet | yes | +47.5% | 1.22 | 1.16 | 2.70 | 1.23 | 1.57 | ok |

(raw numbers: C:\Users\Odwitiyo\Desktop\Topological Self Regulation\results\training_compute.json)
