from imports import *

BCELoss = torch.nn.BCELoss(reduction="sum")

def loss_function(recon_x, x, mu, log_var, beta):
    bce = BCELoss(recon_x, x)
    kld = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    #print(f"bce: {bce}, kld: {kld}")
    return bce + kld


def train(in_shape, max_epochs, latent_size, learning_rate, beta, train_loader, test_loader, device):
    model = VarAutoEncoder(
        spatial_dims=2,
        in_shape=in_shape,
        out_channels=1,
        latent_size=latent_size,
        channels=(16, 32, 64, 128, 256),
        strides=(1, 2, 2, 2, 2),
        # channels=(16, 32, 64), #change in generate images too
        # strides=(1, 2, 2),
    ).to(device)

    # Create optimiser
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    avg_train_losses = []
    test_losses = []

    #t = trange(max_epochs, leave=True, desc="epoch 0, average train loss: ?, test loss: ?")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        for batch_data in train_loader:
            inputs = batch_data["im"].to(device)
            optimizer.zero_grad()

            recon_batch, mu, log_var, _ = model(inputs)
            loss = loss_function(recon_batch, inputs, mu, log_var, beta)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_losses.append(epoch_loss / len(train_loader.dataset))

        # Test
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_data in test_loader:
                inputs = batch_data["im"].to(device)
                recon, mu, log_var, _ = model(inputs)
                # sum up batch loss
                test_loss += loss_function(recon, inputs, mu, log_var, beta).item()
        test_losses.append(test_loss / len(test_loader.dataset))
        print(f"epoch {epoch + 1}, average train loss: " f"{avg_train_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}")
        # t.set_description(
        #     f"epoch {epoch + 1}, average train loss: " f"{avg_train_losses[-1]:.4f}, test loss: {test_losses[-1]:.4f}"
        # )
    return model, avg_train_losses, test_losses