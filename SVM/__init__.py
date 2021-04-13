from SVM.svm import SVM_test

if __name__ == '__main__':
    # SVM_test(k=8, fun_name='classical', data_name='ar',
    #          C=0.3, toler=0.1, max_iter=3, kernel_type='rbf', sigma=0.55,
    #          n_used_class=10, PCA_dimension=None, LDA_dimension=119)
    # c=0.2 LDA 0.929
    SVM_test(k=5, fun_name='classical', data_name='iris',
             C=0.3, toler=0.1, max_iter=3, kernel_type='rbf', sigma=0.55,
             n_used_class=10, PCA_dimension=None, LDA_dimension=None)