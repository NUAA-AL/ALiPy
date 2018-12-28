from acepy.data_manipulate import split_multi_label

# 3 instances with 3 labels.
mult_y = [[1, 1, 1], [0, 1, 1], [0, 1, 0]]  
train_idx, test_idx, label_idx, unlabel_idx = split_multi_label(
    y=mult_y, split_count=1, all_class=False,
    test_ratio=0.3, initial_label_rate=0.5,
    saving_path=None
)

print(train_idx)
print(test_idx)
print(label_idx)
print(unlabel_idx)