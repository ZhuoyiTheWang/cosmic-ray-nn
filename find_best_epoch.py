def read_loss_data(folder_name, model_num):
    """Read loss data from a file with two columns: loss and val_loss."""
    losses = []
    val_losses = []
    with open(f'home/zwang/cosmic-ray-nn/training/{folder_name}/training_history{model_num}.txt', 'r') as file:
        for line in file:
            loss, val_loss = line.strip().split()
            losses.append(float(loss))
            val_losses.append(float(val_loss))
    return losses, val_losses

def find_early_stopping_point(val_loss, patience):
    for i in range(0, len(val_loss) - 20):
        new_minimum_found = False
        for j in range(1, patience + 1):
            if (val_loss[i + j] < val_loss[i]):
                new_minimum_found = True
                continue
        if (new_minimum_found == False):
            print(f'Training terminates at {i + 1} epochs with a patience of {patience}')
            return i
        

            
        

file_name = input("Please enter the folder of the text file containing the loss and validation loss history: ")
model_num = input("Please enter the model number that you would like to view: ")
loss, val_loss = read_loss_data(file_name, model_num)

patience = input("Please enter the patience number for finding program termination: ")
find_early_stopping_point(val_loss, int(patience))