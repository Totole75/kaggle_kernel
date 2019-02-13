import tools
import kernel

folder_name = 'data'
file_feat = 'Xtr1_mat100.csv'
feat = tools.read_file(os.path.join(folder_name, file_feat))
file_label = 'Ytr1.csv'
label = tools.read_file(os.path.join(folder_name, file_label))

K = kernel.gaussian_kernel(feat, 1)