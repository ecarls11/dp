import matplotlib.pyplot as plt 
import csv

train_accs = []
train_losses = []

with open('logtest.csv','r') as csvfile:
    next(csvfile)
    rows = csv.reader(csvfile, delimiter=',')
    for row in rows:
        train_accs.append(int(row[0]))
        train_losses.append(int(row[1]))

acc, = plt.plot(train_accs, 'r', label = 'Training Accuracy')
loss, = plt.plot(train_losses, 'b', label = 'Training Loss')
print("Max train accuracy: ", max(train_accs))
print("Max validation accuracy: ", max(train_losses))
print("Training accuracy is red and training loss is blue. ")

#plt.plot(x,y, label='Loaded from file!')
plt.xlabel('Steps (100s)')
plt.ylabel('Accuracy, Loss')
plt.title('Training Accuracy and Loss')

plt.legend(handles=[acc, loss])

plt.legend()
plt.show()