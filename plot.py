import matplotlib.pyplot as plt 
import csv
import numpy as np

train_accs = np.zeros(1)
train_losses = np.zeros(1)
print(train_accs.shape)

with open('../logs/val_vgg.csv','r') as f:
    count = -1
    for line in f:
        # count += 1
        # if count % 10 == 0:
            acc, loss = line.split()
            acc = float(acc)
            loss = float(loss)
            train_accs = np.append(train_accs, acc)
            train_losses = np.append(train_losses, loss)
        # train_accs.append(acc)
        # train_losses.append(loss)

    train_accs = train_accs[1:]
    train_losses = train_losses[1:]

    steps = list(range(len(train_accs)))
    for i in range(len(steps)):
        steps[i] *= 1#100
    steps = np.asarray(steps)

    # next(csvfile)
    # rows = csv.reader(csvfile, delimiter=',')
    # for row in rows:
    #     print(row)
    #     train_accs.append(float(row[0][0]))
    #     train_losses.append(float(row[0][1]))

plt.figure(1)
plt.plot(steps, train_accs, 'r', label = 'Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Mini-VGG Validation Accuracy')
# plt.legend(handles=[acc])

plt.figure(2)
plt.plot(steps, train_losses, 'b', label = 'Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Mini-VGG Validation Loss')
# plt.legend(handles=[loss])

print("Max train accuracy: ", max(train_accs))
print("Min train loss: ", min(train_losses))
print("Training accuracy is red and training loss is blue. ")

#plt.plot(x,y, label='Loaded from file!')


# plt.legend(handles=[acc, loss])

# plt.legend()
plt.show()