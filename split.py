csvfile = open('creditcard_data.csv', 'r').readlines()
num_fraud = 0
fraud = open("fraud.csv", 'w+')
notfraud = open("notfraud.csv", 'w+')
feature_labels = "A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z,AA,AB,AC,AD\n"

# Write all the fraudulent data to fraud.csv
for i in range(len(csvfile)):
    if int(csvfile[i][-2:-1]) == 1:
        num_fraud += 1
        fraud.writelines(csvfile[i])

# Write the same number of non-fraudulent data to notfraud.csv
for i in range(num_fraud):
    if int(csvfile[i][-2:-1]) == 0:
        notfraud.writelines(csvfile[i])

fraud.close()
notfraud.close()
test = open("test.csv", 'w+')
training = open("training.csv", 'w+')
fraud = open('fraud.csv', 'r').readlines()
notfraud = open('notfraud.csv', 'r').readlines()

num_training = 0.8 * num_fraud
num_test = 1 - num_training

training.writelines(feature_labels)
test.writelines(feature_labels)

# Write 80% of data to training and 20% to test (alternating between fraud and non-fraud data)
for i in range(1, len(fraud)):
    if i <= num_training:
        training.writelines(fraud[i])
        training.writelines(notfraud[i])
    else:
        test.writelines(fraud[i])
        test.writelines(notfraud[i])
