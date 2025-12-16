# Dual-clone networks with collaborative optimization for data-free robust stealing attacks

# Attack

#Due to the need for anonymity, I deleted the code for downloading the standard dataset CIFAR-10/100 and only retained the core code.

To perform DNCO and train the clone model, you can run the following commands with default configuration:

- For CIFAR-10:
  ```bash
  python train.py \
  --arch ResNet18 \
  --target_arch ResNet18 \
  --target_defense AT \
  --data CIFAR10 \
  --batch_size 256 \
  --epoch 300 \
  --lr 0.1 \
  --N_C 500 \
  --N_G 10 \
  --lr_G 0.002 \
  --lr_z 0.01 \
  --lr_hee 0.03 \
  --steps_hee 10 \
  --query_mode hee \
  --label_smooth_factor 0.2 \
  --lam 3 \
  --result_dir results
  ```

- For CIFAR-100:

  ```
  python train.py \
  --arch ResNet18 \
  --target_arch ResNet18 \
  --target_defense AT \
  --data CIFAR100 \
  --batch_size 512 \
  --epoch 300 \
  --lr 0.1 \
  --N_C 500 \
  --N_G 15 \
  --lr_G 0.005 \
  --lr_z 0.015 \
  --lr_hee 0.03 \
  --steps_hee 10 \
  --query_mode hee \
  --label_smooth_factor 0.02 \
  --lam 3 \
  --result_dir results
  ```


