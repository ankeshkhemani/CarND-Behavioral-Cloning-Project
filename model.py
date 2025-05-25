def get_random_row(data_df, data_count):
    return data_df.iloc[np.random.randint(data_count-1)]

def flip_image(image, steering):
    if random.random() >= .5 and abs(steering) > 0.1:
        image = cv2.flip(image, 1)
        steering = -steering
    return image, steering

def crop_if_needed(image, crop_image):
    if crop_image:
        image = crop_camera(image)
    return image

def create_batch(data_df, log_path, cameras, crop_image, batch_size):
    data_count = len(data_df)
    features = []
    labels = []

    while len(features) < batch_size:
        row = get_random_row(data_df, data_count)

        image, steering = jitter_camera_image(row, log_path, cameras)
        image, steering = flip_image(image, steering)
        image = crop_if_needed(image, crop_image)

        features.append(image)
        labels.append(steering)
    return features, labels

def gen_train_data(log_path='./data', log_file='driving_log.csv', skiprows=1,
                   cameras=cameras, filter_straights=False,
                   crop_image=True, batch_size=128):

    print("Cameras: ", cameras)
    print("Log path: ", log_path)
    print("Log file: ", log_file)

    column_names = ['center', 'left', 'right','steering', 'throttle', 'brake', 'speed']
    data_df = pd.read_csv(log_path+'/'+log_file, names=column_names, skiprows=skiprows)

    if filter_straights:
        data_df = filter_driving_straight(data_df)

    print("Log with %d rows." % (len(data_df)))

    while True:
        features, labels = create_batch(data_df, log_path, cameras, crop_image, batch_size)
        yield (np.array(features), np.array(labels))