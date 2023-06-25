from imports import *

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### data
    im_dir = "C:/Users/olive/OneDrive/Desktop/Liver Files/sliced-data/Images/"
    all_filenames = [os.path.join(im_dir, filename) for filename in os.listdir(im_dir)]
    test_frac = 0.2
    num_test = 2#int(len(all_filenames) * test_frac)
    num_train = 1#len(all_filenames) - num_test
    train_datadict = [{"im": fname} for fname in all_filenames[:num_train]]
    test_datadict = [{"im": fname} for fname in all_filenames[-num_test:]]
    print(f"total number of images: {len(all_filenames)}")
    print(f"number of images for training: {len(train_datadict)}")
    print(f"number of images for testing: {len(test_datadict)}")

    ### dataset and loader

    batch_size = 2
    num_workers = 1

    from transforms import train_transforms, test_transforms

    train_ds = CacheDataset(train_datadict, train_transforms, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_ds = CacheDataset(test_datadict, test_transforms, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ### training
    from train_loop import train

    max_epochs = 50
    learning_rate = 1e-4
    beta = 100  # KL beta weighting. increase for disentangled VAE
    latent_size = 2
    # VAE constructor needs image shape
    im_shape = train_transforms(train_datadict[0])["im"].shape
    model, avg_train_losses, test_losses = train(im_shape, 
                                                 max_epochs, 
                                                 latent_size, 
                                                 learning_rate, 
                                                 beta,
                                                 train_loader,
                                                 test_loader,
                                                 device)

    plt.figure()
    plt.title("Epoch losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    for y, label in zip([avg_train_losses, test_losses], ["avg train loss", "test loss"]):
        x = list(range(1, len(y) + 1))
        (line,) = plt.plot(x, y)
        line.set_label(label)
    plt.legend()


    ## scatter distribution
    for j, loader in enumerate([train_loader, test_loader]):
        for i, batch_data in enumerate(loader):
            inputs = batch_data["im"].to(device)
            o = model.reparameterize(*model.encode_forward(inputs)).detach().cpu().numpy()
            if i + j == 0:
                latent_coords = o
            else:
                np.vstack((latent_coords, o))

    if latent_size < 4:
        fig = plt.figure()
        if latent_size == 2:
            plt.scatter(latent_coords[:, 0], latent_coords[:, 1], c="r", marker="o")
        elif latent_size == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(latent_coords[:, 0], latent_coords[:, 1], latent_coords[:, 2], c="r", marker="o")
            ax.set_xlabel("dim 1")
            ax.set_ylabel("dim 2")
            ax.set_zlabel("dim 3")
        plt.show()


    ## viewing
    num_ims = 3
    out = [[[] for _ in range(num_ims)] for _ in range(latent_size - 1)]
    dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
    model.eval()
    with torch.no_grad():
        for z in range(latent_size - 1):
            for z in range(latent_size - 1):
                for y, j in enumerate(torch.linspace(0.05, 0.95, num_ims)):
                    for i in torch.linspace(0.05, 0.95, num_ims):
                        sample = torch.zeros(1, latent_size).to(device)
                        sample[0, z] = dist.icdf(j)
                        sample[0, z + 1] = dist.icdf(i)
                        o = model.decode_forward(sample)
                        o = o.detach().cpu().numpy().reshape(im_shape[1:])
                        out[z][y].append(o)

    slices = np.block(out)

    plt.figure(figsize=(20, 12))
    for i in range(slices.shape[0]):
        plt.imshow(slices[i])
        plt.title(f"slice through dims {i} and {i+1} (through centre of other dims)")
        if slices.shape[0] > 1:
            display.clear_output(wait=True)
            display.display(plt.gcf())
            time.sleep(0.1)
    plt.show()