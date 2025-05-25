def gen_train_data(data, labels, batch_size):
    X, y = [], []
    for i in range(batch_size):
        idx = i % len(data)
        X.append(max(0, data[idx]))
        y.append(0 if data[idx] <= 0 and labels[idx] == 0 else 1)
    return X, y