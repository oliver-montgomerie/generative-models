from imports import *
#todo: info file for beta weighting and network shape and transforms and stuff ??

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### data
    #im_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files/sliced-data/Images/"
    im_dir = "/home/omo23/Documents/tumor-patches-data/Images"
    base_save_path = "/home/omo23/Documents/generative-models/VAE-GAN-models"
    all_filenames = [os.path.join(im_dir, filename) for filename in os.listdir(im_dir)]
    test_frac = 0.2
    num_test = int(len(all_filenames) * test_frac) 
    num_train = len(all_filenames) - num_test       
    train_datadict = [{"im": fname} for fname in all_filenames[:num_train]]
    test_datadict = [{"im": fname} for fname in all_filenames[-num_test:]]
    print(f"total number of images: {len(all_filenames)}")
    print(f"number of images for training: {len(train_datadict)}")
    print(f"number of images for testing: {len(test_datadict)}")

    ### dataset and loader
    batch_size = 16
    num_workers = 8

    from transforms import load_tumor_transforms

    train_ds = CacheDataset(train_datadict, load_tumor_transforms, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_list_data_collate)
    test_ds = CacheDataset(test_datadict, load_tumor_transforms, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=pad_list_data_collate)

    ### training
    from vae_gan_train_loop import train

    max_epochs = 30
    learning_rate = 1e-4
    beta = 1  # KL beta weighting. increase for disentangled VAE    
    latent_sizes = [5, 10]
    for latent_size in latent_sizes:
        save_path = os.path.join(base_save_path, "latent-"+str(latent_size)+"-epochs-"+str(max_epochs))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print("Created folder:",save_path)
        else:
            print(f"Overwriting {save_path}")

        # VAE constructor needs image shape
        im_shape = load_tumor_transforms(train_datadict[0])["im"].shape
        avg_train_losses, disc_losses, test_losses = train(im_shape, 
                                                    max_epochs, 
                                                    latent_size, 
                                                    learning_rate, 
                                                    beta,
                                                    train_loader,
                                                    test_loader,
                                                    device,
                                                    save_path)
        

        plt.figure()
        plt.title("Epoch losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        for y, label in zip([avg_train_losses, test_losses], ["avg train loss", "test loss"]):
            x = list(range(1, len(y) + 1))
            (line,) = plt.plot(x, y)
            line.set_label(label)
        plt.legend()
        plt.savefig(os.path.join(save_path, "train_test_loss.png"), bbox_inches='tight')
        plt.close()

        plt.figure()
        plt.title("Disc losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        for y, label in zip([disc_losses], ["discriminator loss"]):
            x = list(range(1, len(y) + 1))
            (line,) = plt.plot(x, y)
            line.set_label(label)
        plt.legend()
        plt.savefig(os.path.join(save_path, "discriminator_loss.png"), bbox_inches='tight')
        plt.close()



    
