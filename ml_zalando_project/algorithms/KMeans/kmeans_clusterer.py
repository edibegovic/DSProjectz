import numpy as np

# filepaths
lda3_projected_path = '/home/gregkoncz/git/DSProjectz/ml_zalando_project/data/lda3_projected.npy'
lda4_projected_path = '/home/gregkoncz/git/DSProjectz/ml_zalando_project/data/lda4_projected.npy'
all_input_path = '/home/gregkoncz/git/DSProjectz/ml_zalando_project/data/fashion_train.npy'
lda3_projector_path = '/home/gregkoncz/git/DSProjectz/ml_zalando_project/algorithms/LDA/sorted_eigvecs_3.npy'
lda4_projector_path = '/home/gregkoncz/git/DSProjectz/ml_zalando_project/algorithms/LDA/sorted_eigvecs_4.npy'
test_set_path = '/home/gregkoncz/git/DSProjectz/ml_zalando_project/data/test_data_no_touch/fashion_test.npy'

def classifier(train_instances, train_labels, test_instances, test_labels):
    means = [np.mean(train_instances[train_labels == i], axis = 0) for i in range(5)]
    print(means)
    accurate = 0
    for idx, row in enumerate(test_instances):
        #print(row)
        distances = np.array([np.linalg.norm(row - this_mean) for this_mean in means])
        #print(distances)
        if (distances.argmin() == test_labels[idx]):
            accurate += 1
    print(accurate/test_labels.shape[0])


if __name__ == '__main__':
    # get all data
    lda3_projected = np.load(lda3_projected_path)
    #print(lda3_projected.shape)
    lda4_projected = np.load(lda4_projected_path)
    #print(lda4_projected.shape)
    all_input = np.load(all_input_path)
    #print(all_input.shape)
    lda3_projector = np.load(lda3_projector_path)
    #print(lda3_projector.shape)
    lda4_projector = np.load(lda4_projector_path)
    test_set = np.load(test_set_path)

    # data manipulations
    np.random.shuffle(all_input)
    labels = all_input[:, -1]
    current_train = all_input[:, :-1]
    d3_projected = np.dot(current_train, lda3_projector)
    d4_projected = np.dot(current_train, lda4_projector)
    
    d3_train = d3_projected[:8000, :]
    d3_train_labels = labels[:8000]
    d3_test = d3_projected[8001:, :]
    d3_test_labels = labels[8001:]

    d4_train = d4_projected[:8000, :]
    d4_train_labels = labels[:8000]
    d4_test = d4_projected[8001:, :]
    d4_test_labels = labels[8001:]

    classifier(d3_train, d3_train_labels, d3_test, d3_test_labels)
    classifier(d4_train, d4_train_labels, d4_test, d4_test_labels)

    # big test set and with all validation

    b_test_labels = test_set[:, -1]
    b_test_input = test_set[:, :-1]
    d3_b_test = np.dot(b_test_input, lda3_projector)
    d4_b_test = np.dot(b_test_input, lda4_projector)

    classifier(d3_projected, labels, d3_b_test, b_test_labels)
    classifier(d4_projected, labels, d4_b_test, b_test_labels)
