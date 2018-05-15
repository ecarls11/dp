import matplotlib.pyplot as plt 
import csv
import numpy as np


learn = ['0.01', '0.001', '0.0001']
for lr in learn:
    save_name = 'vgg_adam' + lr
    train_accs = np.zeros(1)
    train_losses = np.zeros(1)
    val_accs = np.zeros(1)
    val_losses = np.zeros(1)

    with open('../logs/' + save_name + '/train.csv','r') as f:
        for line in f:
            acc, loss = line.split()
            acc = float(acc)
            loss = float(loss)
            train_accs = np.append(train_accs, acc)
            train_losses = np.append(train_losses, loss)

        train_accs = train_accs[1:]
        train_losses = train_losses[1:]

        steps = list(range(len(train_accs)))
        for i in range(len(steps)):
            steps[i] *= 500
        steps = np.asarray(steps)


    with open('../logs/' + save_name + '/val.csv','r') as f:
        count = -1
        for line in f:
            acc, loss = line.split()
            acc = float(acc)
            loss = float(loss)
            val_accs = np.append(val_accs, acc)
            val_losses = np.append(val_losses, loss)

        val_accs = val_accs[1:]
        val_losses = val_losses[1:]

        steps = list(range(len(val_accs)))
        for i in range(len(steps)):
            steps[i] *= 500
        steps = np.asarray(steps)

    plt.figure(1)
    plt.plot(steps, train_accs, 'r', label='Training Accuracy')
    plt.plot(steps, val_accs, 'b', label='Validation Accuracy')
    plt.xlabel('Step')
    plt.ylabel('Accuracy')
    plt.title(save_name + '_acc.png')
    plt.legend()
    plt.savefig('pics/combined/' + save_name + '_acc.png')
    plt.close(1)

    plt.figure(2)
    plt.plot(steps, train_losses, 'r', label='Training Loss')
    plt.plot(steps, val_losses, 'b', label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title(save_name + '_loss.png')
    plt.legend()
    plt.savefig('pics/combined/' + save_name + '_loss.png')
    plt.close(2)

    print("Max train accuracy: ", max(train_accs))
    print("Min train loss: ", min(train_losses))
    print("Training accuracy is red and training loss is blue. ")

    

#plt.plot(x,y, label='Loaded from file!')


# plt.legend(handles=[acc, loss])

# plt.legend()
# plt.show()