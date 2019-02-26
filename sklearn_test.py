from sklearn import svm

file_idx = 0

folder_name = 'data'
train_file_name = "Xtr" + str(file_idx) + ".csv"
result_file_name = "Ytr"+ str(file_idx) + ".csv"

training_percentage = 0.15
data_array = tools.read_file(os.path.join(folder_name, train_file_name))
result_array = tools.read_file(os.path.join(folder_name, result_file_name))
training_nb = round(training_percentage*len(data_array))
training_array = data_array[:training_nb, :]
training_labels = result_array[:training_nb]
test_array = data_array[result_array.shape[0]-training_nb:, :]
test_labels = result_array[result_array.shape[0]-training_nb:]


name_file_training = 'lev_dist_training_array' + str(file_idx)
lev_dist_training_array = np.load(name_file_training + '.npy')

name_file_test = 'lev_dist_test_array' + str(file_idx)
lev_dist_test_array = np.load(name_file_test + '.npy')


clf = svm.SVC(kernel='precomputed')

clf.fit(lev_dist_training_array, training_labels)

predictions = clf.predict(lev_dist_test_array)

test_classes = []
for i in range(predictions.shape[0]):
    test_classes.append(0 if predictions[i] < 0 else 1)
    
success_percentage = np.mean(np.equal(test_classes, ((test_labels+1)/2).astype(int)))
print("Result for file " + str(file_idx) + " : " + str(success_percentage))
