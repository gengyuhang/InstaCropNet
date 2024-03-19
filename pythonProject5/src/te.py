import scipy.io
#Validation_Gt = scipy.io.loadmat('/Users/gengyuhang/PycharmProjects/pythonProject5/data/SIDD_Small_sRGB_Only/ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb']
#print((Validation_Gt.shape))
Validation_Gt = scipy.io.loadmat('/Users/gengyuhang/Desktop/BSD68_gs15.mat')['data'][0][0]
print((Validation_Gt.shape))
