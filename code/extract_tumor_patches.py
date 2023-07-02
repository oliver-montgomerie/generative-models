from imports import *

def stuff():
    img_save_path = "/home/omo23/Documents/tumor-patches-data/Images"
    lbl_save_path = "/home/omo23/Documents/tumor-patches-data/Labels"
    
    #Data loading
    data_dir = "/home/omo23/Documents/sliced-data"
    all_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
    all_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]
    # filter FOR slices with small tumor area or no tumor
    data_dicts = [item for item in data_dicts if file_tumor_size(item) < min_tumor_size]

    test_files, val_files, train_files = [], [], []
    for d in data_dicts:
        d_num = d['image']
        d_num = d_num[d_num.rfind("/")+1:d_num.rfind("-")] 
        #if d_num in test_files_nums:
        #    test_files.append(d)
        if d_num in val_files_nums:
            val_files.append(d)
        if d_num in train_files_nums:
            train_files.append(d)

    for file in train_files + val_files:
        img_path = file['image']
        lbl_path = file['label']
        fname = img_path[img_path.rfind("/")+1:img_path.rfind(".")] 

        img = nib.load(img_path)
        lbl = nib.load(lbl_path)

        np_img = np.array(img.dataobj)
        np_lbl = np.array(lbl.dataobj)

        np_lbl[np_lbl == 1] = 0 #remove liver label
        gt_seperated_tumor_labels, gt_num_regions = seperate_instances(label_image = np_lbl, background=0, return_num=True, connectivity=None)
        
        for i in range(gt_num_regions):
            # get min x,y and max x,y from each seperate instance
            idx = np.argwhere(gt_seperated_tumor_labels == i+1) 
            tumor_size = idx.shape[0]
            min_0 = np.min(idx[:,0])
            max_0 = np.max(idx[:,0])
            min_1 = np.min(idx[:,1])
            max_1 = np.max(idx[:,1])

            if max_0 - min_0 > largest_0:
                largest_0 = max_0 - min_0
                
            if max_1 - min_1 > largest_1:
                largest_1 = max_1 - min_1


            buffer = 1
            tumor_img = np.copy(np_img[min_0-buffer:max_0+buffer+1, min_1-buffer:max_1+buffer+1])
            tumor_lbl = np.copy(gt_seperated_tumor_labels[min_0-buffer:max_0+buffer+1, min_1-buffer:max_1+buffer+1])

            tumor_img[tumor_lbl != i+1] = np.min(np_img)
            tumor_lbl[tumor_lbl != i+1] = 0.0
            tumor_lbl[tumor_lbl == i+1] = 1.0

            # plt.figure()
            # plt.subplot(2,2,1)
            # plt.imshow(np_img,cmap="gray")
            # plt.subplot(2,2,2)
            # plt.imshow(np_lbl)
            # plt.subplot(2,2,3)
            # plt.imshow(tumor_img,cmap="gray")
            # plt.subplot(2,2,4)
            # plt.imshow(tumor_lbl)
            # plt.show()

            ni_img = nib.Nifti1Image(tumor_img, img.affine) #, img.header (?)
            ni_lbl = nib.Nifti1Image(tumor_lbl, lbl.affine, dtype='<i2')
            #print(f"saving: {img_path[:-4]}_{str(tumor_size)}.nii")
            nib.save(ni_img, os.path.join(img_save_path, fname + "_" + str(tumor_size) + ".nii"))
            ##nib.save(ni_lbl, os.path.join(lbl_save_path, fname + "_" + str(tumor_size) + ".nii"))

    print("largest: ",largest_0, largest_1)


if __name__ == '__main__':
    stuff()