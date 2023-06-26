# lr
lr0: 0.0001
warmup_lr: 0.00001   
warm_epoch:1


# setting
num_classes: 2  # 绝缘子串 
                # II02: 标准串
                # others: 模糊、不易辨识串

# training
epochs: 1000
batch_size: 4
save_interval: 20
test_interval: 1
