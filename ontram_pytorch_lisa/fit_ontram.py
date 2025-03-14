import torch
import numpy as np

from .loss import ontram_nll

def fit_ontram(model, train_loader, val_loader=None, checkpoint_path=None, optimizer=None, use_scheduler=False, epochs=10, si=True):
    """
    Fit function uniting simple intercept and complex intercept terms 
    
    Args:
        model: model of class OntramModel
        train_loader: torch train loader containing either (tabular_data, outcome) or (tabular_data, image_data, outcome)
        val_loader: same as train loader but with validation data
        checkpoint_path: Path where to store the model weights
        optimizer: by default Adam. But any optimizer can be provided
        lr: learning rate
        epochs: number of epochs
        devive: "cpu" or "cuda"
    """
    model = model
    train_loader = train_loader
    val_loader = val_loader
    checkpoint_path = checkpoint_path
    epochs = epochs
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-7, weight_decay=0.0)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Train with GPU support.")
    else:
        device = torch.device('cpu')
        print("No GPU found, train with CPU support.")

    def fit_ontram_si(model, train_loader, val_loader, checkpoint_path, optimizer, use_scheduler, epochs, device):
        """
        Training process for ontrams with a simple intercept and up to two shift terms. 

        Args:
            model: model of class OntramModel
            train_loader: torch train loader containing either (tabular_data, outcome) or (tabular_data, image_data, outcome)
            val_loader: same as train loader but with validation data
            checkpoint_path: Path where to store the model weights
            optimizer: by default Adam. But any optimizer can be provided
            lr: learning rate
            epochs: number of epochs
            devive: "cpu" or "cuda"
        """

        # Define parameters
        best_val_loss = float('inf')
        train_loss_hist = []
        val_loss_hist = []

        # model to device
        model.to(device)

        # No shift term -----------------------------------------------------------------------
        if model.nn_shift is None:
            for epoch in range(epochs):
                # Training phase -------------------
                model.train()
                train_loss = 0.0
                for y_tr in train_loader:
                    y_tr = y_tr[0]
                    y_tr = y_tr.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    int_tr = torch.from_numpy(np.ones(shape=[len(y_tr),1])).float()
                    int_tr = int_tr.to(device)
                    y_pred = model(int_tr)
                    loss = ontram_nll(y_pred, y_tr)

                    # Backward pass
                    loss.backward() 
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss/len(train_loader)
                train_loss_hist.append(avg_train_loss)

                # Validation phase -------------------
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    # No gradient computation
                    with torch.no_grad():
                        for y_val in val_loader:
                            y_val = y_val[0]
                            y_val = y_val.to(device)

                            # Forward pass
                            int_val = torch.from_numpy(np.ones(shape=[len(y_val),1])).float()
                            int_val = int_val.to(device)
                            y_pred = model(int_val) 
                            loss = ontram_nll(y_pred, y_val)

                            val_loss += loss.item()

                    avg_val_loss = val_loss/len(val_loader)
                    val_loss_hist.append(avg_val_loss)

                    # Save model
                    if checkpoint_path is not None:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save({'epoch': epoch,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'val_loss': avg_val_loss,
                                        },
                                        checkpoint_path)
                else:
                    avg_val_loss = 0.0

                # Scheduler
                if use_scheduler:
                    scheduler.step()
                    print(f"New lr: {scheduler.get_last_lr()[0]}")
                
                # Epoch output ------------------------
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}"
                )


        # One shift term -----------------------------------------------------------------------
        elif len(model.nn_shift)==1:
            for epoch in range(epochs):
                # Training phase -------------------
                model.train()
                train_loss = 0.0
                for x_tr, y_tr in train_loader:
                    x_tr, y_tr = x_tr.to(device), y_tr.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    int_tr = torch.from_numpy(np.ones(shape=[len(y_tr),1])).float()
                    int_tr = int_tr.to(device)
                    y_pred = model(int_tr, [x_tr])
                    loss = ontram_nll(y_pred, y_tr)

                    # Backward pass
                    loss.backward() 
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss/len(train_loader)
                train_loss_hist.append(avg_train_loss)
                
                # Validation phase -------------------
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    # No gradient computation
                    with torch.no_grad():
                        for x_val, y_val in val_loader:
                            x_val, y_val = x_val.to(device), y_val.to(device)

                            # Forward pass
                            int_val = torch.from_numpy(np.ones(shape=[len(y_val),1])).float()
                            int_val = int_val.to(device)
                            y_pred = model(int_val, [x_val])
                            loss = ontram_nll(y_pred, y_val)

                            val_loss += loss.item()

                    avg_val_loss = val_loss/len(val_loader)
                    val_loss_hist.append(avg_val_loss)

                    # Save model
                    if checkpoint_path is not None:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save({'epoch': epoch,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'val_loss': avg_val_loss,
                                        },
                                        checkpoint_path)
                else:
                    avg_val_loss = 0.0

                # Scheduler
                if use_scheduler:
                    scheduler.step()
                    print(f"New lr: {scheduler.get_last_lr()[0]}")
                
                # Epoch output ------------------------
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}"
                )

        # Two shift terms -----------------------------------------------------------------------
        elif len(model.nn_shift)==2:
            for epoch in range(epochs):
                # Training phase -----------------------
                model.train()
                train_loss = 0.0
                for x_tr1, x_tr2, y_tr in train_loader:
                    x_tr1, x_tr2, y_tr = x_tr1.to(device), x_tr2.to(device), y_tr.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    int_tr = torch.from_numpy(np.ones(shape=[len(y_tr),1])).float()
                    int_tr = int_tr.to(device)
                    y_pred = model(int_tr, [x_tr1, x_tr2])
                    loss = ontram_nll(y_pred, y_tr)

                    # Backward pass
                    loss.backward() 
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss/len(train_loader)
                train_loss_hist.append(avg_train_loss)
                
                # Validation phase ---------------------
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    # No gradient computation
                    with torch.no_grad():
                        for x_val1, x_val2, y_val in val_loader:
                            x_val1, x_val2, y_val = x_val1.to(device), x_val2.to(device), y_val.to(device)

                            # Forward pass
                            int_val = torch.from_numpy(np.ones(shape=[len(y_val),1])).float()
                            int_val = int_val.to(device)
                            y_pred = model(int_val, [x_val1, x_val2])
                            loss = ontram_nll(y_pred, y_val)

                            val_loss += loss.item()

                    avg_val_loss = val_loss/len(val_loader)
                    val_loss_hist.append(avg_val_loss)

                    # Save model
                    if checkpoint_path is not None:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save({'epoch': epoch,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'val_loss': avg_val_loss,
                                        },
                                        checkpoint_path)
                else:
                    avg_val_loss = 0.0

                # Scheduler
                if use_scheduler:
                    scheduler.step()
                    print(f"New lr: {scheduler.get_last_lr()[0]}")
                
                # Epoch output ------------------------
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}"
                )

        # Three shift terms -----------------------------------------------------------------------
        elif len(model.nn_shift)==3:
            for epoch in range(epochs):
                # Training phase -----------------------
                model.train()
                train_loss = 0.0
                for x_tr1, x_tr2, x_tr3, y_tr in train_loader:
                    x_tr1, x_tr2, x_tr3, y_tr = x_tr1.to(device), x_tr2.to(device), x_tr3.to(device), y_tr.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    int_tr = torch.from_numpy(np.ones(shape=[len(y_tr),1])).float()
                    int_tr = int_tr.to(device)
                    y_pred = model(int_tr, [x_tr1, x_tr2, x_tr3])
                    loss = ontram_nll(y_pred, y_tr)

                    # Backward pass
                    loss.backward() 
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss/len(train_loader)
                train_loss_hist.append(avg_train_loss)
                
                # Validation phase ---------------------
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    # No gradient computation
                    with torch.no_grad():
                        for x_val1, x_val2, x_val3, y_val in val_loader:
                            x_val1, x_val2, x_val3, y_val = x_val1.to(device), x_val2.to(device), x_val3.to(device), y_val.to(device)

                            # Forward pass
                            int_val = torch.from_numpy(np.ones(shape=[len(y_val),1])).float()
                            int_val = int_val.to(device)
                            y_pred = model(int_val, [x_val1, x_val2, x_val3])
                            loss = ontram_nll(y_pred, y_val)

                            val_loss += loss.item()

                    avg_val_loss = val_loss/len(val_loader)
                    val_loss_hist.append(avg_val_loss)

                    # Save model
                    if checkpoint_path is not None:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save({'epoch': epoch,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'val_loss': avg_val_loss,
                                        },
                                        checkpoint_path)
                else:
                    avg_val_loss = 0.0

                # Scheduler
                if use_scheduler:
                    scheduler.step()
                    print(f"New lr: {scheduler.get_last_lr()[0]}")
                
                # Epoch output ------------------------
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}"
                )

        # More than three data modalities ----------------------------------------------------
        elif len(model.nn_shift)>3:
            raise ValueError("More than three data modalities provided.")
        
        # Return -----------------------------------------------------------------------------

        return {'train_loss': train_loss_hist, 'val_loss': val_loss_hist}
    

    def fit_ontram_ci(model, train_loader, val_loader, checkpoint_path, optimizer, use_scheduler, epochs, device):
        """
        Training process for ontrams with a complex intercept and up to two shift terms. 

        Args:
            model: model of class OntramModel
            train_loader: torch train loader containing either (tabular_data, outcome) or (tabular_data, image_data, outcome)
            val_loader: same as train loader but with validation data
            checkpoint_path: Path where to store the model weights
            optimizer: by default Adam. but any optimizer can be provided
            lr: learning rate
            epochs: number of epochs
            devive: "cpu" or "cuda" 
        """

        # Define parameters
        best_val_loss = float('inf')
        train_loss_hist = []
        val_loss_hist = []

        # Learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
        # model to device
        model.to(device)

        # No shift term -----------------------------------------------------------------------
        if model.nn_shift is None:
            for epoch in range(epochs):
                # Training phase -------------------
                model.train()
                train_loss = 0.0
                for x_tr, y_tr in train_loader:
                    x_tr, y_tr = x_tr.to(device), y_tr.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    y_pred = model(x_tr)
                    loss = ontram_nll(y_pred, y_tr)
                    # print("Batch loss: ", loss)

                    # Backward pass
                    loss.backward() 
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss/len(train_loader)
                train_loss_hist.append(avg_train_loss)
                
                # Validation phase -------------------
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    # No gradient computation
                    with torch.no_grad():
                        for x_val, y_val in val_loader:
                            x_val, y_val = x_val.to(device), y_val.to(device)

                            # Forward pass
                            y_pred = model(x_val)  
                            loss = ontram_nll(y_pred, y_val)

                            val_loss += loss.item()

                    avg_val_loss = val_loss/len(val_loader)
                    val_loss_hist.append(avg_val_loss)

                    # Save model
                    if checkpoint_path is not None:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save({'epoch': epoch,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'val_loss': avg_val_loss,
                                        },
                                        checkpoint_path)
                else:
                    avg_val_loss = 0.0

                # Scheduler
                if use_scheduler:
                    scheduler.step()
                    print(f"New lr: {scheduler.get_last_lr()[0]}")
                
                # Epoch output ------------------------
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}"
                )

        # One shift term -----------------------------------------------------------------------
        elif len(model.nn_shift)==1:
            for epoch in range(epochs):
                # Training phase -----------------------
                model.train()
                train_loss = 0.0
                for x_tr1, x_tr2, y_tr in train_loader:
                    x_tr1, x_tr2, y_tr = x_tr1.to(device), x_tr2.to(device), y_tr.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    y_pred = model(x_tr1, [x_tr2])
                    loss = ontram_nll(y_pred, y_tr)

                    # Backward pass
                    loss.backward() 
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss/len(train_loader)
                train_loss_hist.append(avg_train_loss)
                
                # Validation phase ---------------------
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    # No gradient computation
                    with torch.no_grad():
                        for x_val1, x_val2, y_val in val_loader:
                            x_val1, x_val2, y_val = x_val1.to(device), x_val2.to(device), y_val.to(device)

                            # Forward pass
                            y_pred = model(x_val1, [x_val2])  
                            loss = ontram_nll(y_pred, y_val)

                            val_loss += loss.item()

                    avg_val_loss = val_loss/len(val_loader)
                    val_loss_hist.append(avg_val_loss)

                    # Save model
                    if checkpoint_path is not None:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save({'epoch': epoch,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'val_loss': avg_val_loss,
                                        },
                                        checkpoint_path)
                else:
                    avg_val_loss = 0.0

                # Scheduler
                if use_scheduler:
                    scheduler.step()
                    print(f"New lr: {scheduler.get_last_lr()[0]}")
                
                # Epoch output ------------------------
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}"
                )

         # Two shift terms -----------------------------------------------------------------------
        elif len(model.nn_shift)==2:
            for epoch in range(epochs):
                # Training phase -----------------------
                model.train()
                train_loss = 0.0
                for x_tr1, x_tr2, x_tr3, y_tr in train_loader:
                    x_tr1, x_tr2, x_tr3, y_tr = x_tr1.to(device), x_tr2.to(device), x_tr3.to(device), y_tr.to(device)

                    optimizer.zero_grad()

                    # Forward pass
                    y_pred = model(x_tr1, [x_tr2, x_tr3])
                    loss = ontram_nll(y_pred, y_tr)

                    # Backward pass
                    loss.backward() 
                    optimizer.step()

                    train_loss += loss.item()

                avg_train_loss = train_loss/len(train_loader)
                train_loss_hist.append(avg_train_loss)
                
                # Validation phase ---------------------
                if val_loader is not None:
                    model.eval()
                    val_loss = 0.0
                    # No gradient computation
                    with torch.no_grad():
                        for x_val1, x_val2, x_val3, y_val in val_loader:
                            x_val1, x_val2, x_val3, y_val = x_val1.to(device), x_val2.to(device), x_val3.to(device), y_val.to(device)

                            # Forward pass
                            y_pred = model(x_val1, [x_val2, x_val3])
                            loss = ontram_nll(y_pred, y_val)

                            val_loss += loss.item()

                    avg_val_loss = val_loss/len(val_loader)
                    val_loss_hist.append(avg_val_loss)

                    # Save model
                    if checkpoint_path is not None:
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            torch.save({'epoch': epoch,
                                        'model_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict(),
                                        'val_loss': avg_val_loss,
                                        },
                                        checkpoint_path)
                else:
                    avg_val_loss = 0.0

                # Scheduler
                if use_scheduler:
                    scheduler.step()
                    print(f"New lr: {scheduler.get_last_lr()[0]}")
                
                # Epoch output ------------------------
                print(
                    f"Epoch {epoch + 1}/{epochs}, "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Validation Loss: {avg_val_loss:.4f}"
                )

        # More than three data modalities ----------------------------------------------------
        elif len(model.nn_shift)>2:
            raise ValueError("More than three data modalities provided.")
        
        # Return -----------------------------------------------------------------------------
        
        return {'train_loss': train_loss_hist, 'val_loss': val_loss_hist}
    
    if si:
        return fit_ontram_si(model, train_loader, val_loader, checkpoint_path, optimizer, use_scheduler, epochs, device)
    else:
        return fit_ontram_ci(model, train_loader, val_loader, checkpoint_path, optimizer, use_scheduler, epochs, device)