from imports import *

def generate_a_tumor(model, dist, tumor_shape, device):
    with torch.no_grad():
        sample = torch.zeros(1, latent_size).to(device)
        for s in range(sample.shape[1]):
            #todo: maybe This should be weighted towards the edges for unusual looking tumors?
            sample[0,s] = dist.icdf((torch.rand(1)*0.9) + 0.05)

        o = model.decode_forward(sample) #get output from latent sample
        o = o.detach().cpu().numpy().reshape(tumor_shape)
        mask = np.zeros(o.shape)    #create mask from thresholding tumor
        thresh = (np.max(o) + np.min(o))/2
        mask[o > thresh] = 1
        #clip the tumor to the mask size
        o[mask != 1] = 0

        #rescale back to hounsfield
        o = ((o*2)-1)*200

        # pad to slice size
        pad_size = np.array([560, 560]) - np.array([o.shape[1],o.shape[2]])
        o = np.pad(o, [(0, 0), (pad_size[0], 0), (pad_size[1], 0)])

        pad_size = np.array([560, 560]) - np.array([mask.shape[1],mask.shape[2]])
        mask = np.pad(mask, [(0, 0), (pad_size[0], 0), (pad_size[1], 0)])

        return o, mask


## viewing
def generate_slices(model, tumor_shape, device):
    img_save_path = "/home/omo23/Documents/generated-data/VAE/Images"
    lbl_save_path = "/home/omo23/Documents/generated-data/VAE/Labels"
    
    #Data loading
    data_dir = "/home/omo23/Documents/sliced-data"
    all_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii")))
    all_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii")))
    data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(all_images, all_labels)]
    # filter FOR slices with small tumor area or no tumor
    data_dicts = [item for item in data_dicts if file_tumor_size(item) < min_tumor_size]
    data_dicts = [item for item in data_dicts if file_liver_size(item) > min_liver_size]

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

    num_workers = 4
    batch_size = 16
    from transforms import load_slice_transforms
    ds = CacheDataset(train_files + val_files, load_slice_transforms, num_workers=num_workers)
    data_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_list_data_collate)

    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))

    for batch_data in data_loader:
        #Slices with no/ or small tumor
        imgs, lbls = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
        for i in range(imgs.shape[0]):
            img = imgs[i,:,:,:].cpu()
            lbl = lbls[i,:,:,:].cpu()

            liver_pix = np.argwhere(lbl == 1)

            max_attempts = 20
            good_placement = False
            for attempt in range(max_attempts):
                tumor_img, tumor_lbl = generate_a_tumor(model, dist, tumor_shape, device)
                
                # make sure its large enough
                tumor_size = len(np.argwhere(tumor_lbl == 1))
                if tumor_size < min_tumor_size:
                    continue

                #insert into liver
                tumor_pix = np.argwhere(tumor_lbl == 1)
                tumor_centre = np.mean(tumor_pix, axis = 0,dtype=int)

                location = random.choice(liver_pix)

                #shift the tumor so that the centre = location.
                tumor_lbl = np.roll(tumor_lbl, location[1] - tumor_centre[1], axis=1)
                tumor_lbl = np.roll(tumor_lbl, location[2] - tumor_centre[2], axis=2)
                tumor_img = np.roll(tumor_img, location[1] - tumor_centre[1], axis=1)
                tumor_img = np.roll(tumor_img, location[2] - tumor_centre[2], axis=2)

                tumor_lbl[tumor_lbl==1] = 3
                # add tumor label to img label. if there are lots of 3's then it means outside liver
                # if lots of 4's then it was inside the liver
                # 5 means on- top of another tumor, which is also ok
                gen_lbl = lbl + tumor_lbl

                not_liver = len(np.argwhere(gen_lbl == 3))
                in_liver = len(np.argwhere(gen_lbl > 3))

                print(f"in_liver ratio: {in_liver / (in_liver + not_liver)}")
                if in_liver / (in_liver + not_liver) > 0.95:
                    good_placement = True
                    break #good

             # Todo: add a probability for repeating the process for adding multiple tumors.
             # look at ratio in other slices and use that??


            if good_placement == False:
                continue
                
            # merge images
            # todo: some blending between the two. interpolate between neighbouring images
            gen_img = img.detach().clone().numpy()
            gen_img[tumor_lbl >= 3] = tumor_img[tumor_lbl >= 3]

            plt.figure("New tumor", (18, 6))
            plt.subplot(2,3,1)
            plt.imshow(img[0,:,:], cmap="gray")
            plt.axis('off')
            plt.subplot(2,3,4)
            plt.imshow(lbl[0,:,:], vmin=0, vmax=5)
            plt.axis('off')

            plt.subplot(2,3,2)
            plt.imshow(tumor_img[0,:,:], cmap="gray")
            plt.axis('off')
            plt.subplot(2,3,5)
            plt.imshow(tumor_lbl[0,:,:], vmin=0, vmax=5)
            plt.axis('off')

            plt.subplot(2,3,3)
            plt.imshow(gen_img[0,:,:], cmap="gray")
            plt.axis('off')
            plt.subplot(2,3,6)
            plt.imshow(gen_lbl[0,:,:], vmin=0, vmax=5)
            plt.axis('off')

            plt.show()

            fname = batch_data['image_meta_dict']['filename_or_obj'][i]
            fname = fname[fname.rfind("/")+1:-4]

            gen_lbl[gen_lbl >= 3] = 2 
            ni_img = nib.Nifti1Image(gen_img, img.affine, img.header) #, img.header (?)
            ni_lbl = nib.Nifti1Image(gen_lbl, lbl.affine, lbl.header, dtype='<i2')

            nib.save(ni_img, os.path.join(img_save_path, fname + "_" + str(tumor_size) + ".nii"))
            nib.save(ni_lbl, os.path.join(lbl_save_path, fname + "_" + str(tumor_size) + ".nii"))


       
    


tumor_shape = [1,256,256]
latent_size = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_path = "/home/omo23/Documents/generative-models/VAE-generated/latent-2-epochs-20"

model = VarAutoEncoder(
    spatial_dims=2,
    in_shape=tumor_shape,
    out_channels=1,
    latent_size=latent_size,
    channels=(16, 32, 64, 128, 256),
    strides=(1, 2, 2, 2, 2),
).to(device)
model.load_state_dict(torch.load(os.path.join(load_path, "trained_model.pth")))
model.eval()

generate_slices(model = model,
                tumor_shape=tumor_shape,
                device=device,)