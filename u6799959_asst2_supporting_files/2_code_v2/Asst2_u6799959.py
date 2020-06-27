'''This file is the script of COMP8420 Assignment 2 for u6799959, before run this file,
please read the README file in the deliverable folder, and also, to run this file,
you'd better using a GPU training mode or it would take a long time.
The structure of data loading and model training of this script takes reference to
pytorch transfer learning tutorial which can be found here
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
If you have any question, please send email to u6799959@anu.edu.au'''


import shutil
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time

import copy

'''Part 1: Use one pair of train-test dataset to do hyperparameter tuning and model selection
  due to limited computational power and GPU performance of my computer'''

'''------------------------------------------------------------------------------'''
'''First, we need to customize our training and testing data folder'''
def get_files_from_folder(path):
    files = os.listdir(path)
    return np.asarray(files)

data_path = '.\\faces-emotion_v2\\Subset For Assignment SFEW'
# investigate what directions in original dataset, should be seven directions containing each face emotion
_, dirs, _ = next(os.walk(data_path))

# create training and test directions to be used later
try:
    os.makedirs('.\\faces-emotion_v2\\train_u6799959')
    os.makedirs('.\\faces-emotion_v2\\test_u6799959')
except OSError:
    pass

train_ratio = 0.8  # we use 80% of data for training
data_counter_per_class = np.zeros((len(dirs)))  # numpy array to store sample number for each class

for i in range(len(dirs)):
    current_path = os.path.join(data_path, dirs[i])
    files = get_files_from_folder(current_path)
    data_counter_per_class[i] = len(files)  # count sample numbers for each class
test_counter = np.round(data_counter_per_class * (1 - train_ratio)) # number of test samples each class

path_to_test_data = '.\\faces-emotion_v2\\test_u6799959'
path_to_train_data = '.\\faces-emotion_v2\\train_u6799959'

# transfers files for testing data for each class
for i in range(len(dirs)):
    path_to_original = os.path.join(data_path, dirs[i])  # original data path for each class
    path_to_save_test = os.path.join(path_to_test_data, dirs[i])  # test data file folder for each class
    path_to_save_train = os.path.join(path_to_train_data, dirs[i])  # train data file folder for each class
    #creates dir
    if not os.path.exists(path_to_save_test):
        os.makedirs(path_to_save_test)
    if not os.path.exists(path_to_save_train):
        os.makedirs(path_to_save_train)

    # get all the files for current original data class
    files = get_files_from_folder(path_to_original)
    np.random.shuffle(files)  # randomly shuffle the files

    # copies data for testing folders
    for j in range(int(test_counter[i])):
        dest_folder = os.path.join(path_to_save_test, files[j])
        sour_folder = os.path.join(path_to_original, files[j])
        shutil.copy(src = sour_folder, dst = dest_folder)

    # copies data for training folders
    for k in range(int(test_counter[i]), len(files)):
        dest_folder = os.path.join(path_to_save_train, files[k])
        sour_folder = os.path.join(path_to_original, files[k])
        shutil.copy(src=sour_folder, dst=dest_folder)

'''------------------------------------------------------------------------------'''
'''Next, we need to load the data by using torch data loaders'''
# First, create data transform process for training and testing data
# Data Crop, Flip, Resize and Normalize for training and testing data
# for each image, there are R-B-G three channels
data_transforms = {
    'train_u6799959': transforms.Compose([
        # implement center crop and resize the image size for training dataset to 299 pixels
        transforms.CenterCrop(size = 576),
        transforms.Resize(size = 299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # see https://pytorch.org/docs/stable/torchvision/transforms.html Normalize for details
        # mean (sequence) ¨C Sequence of means for each channel.
        # std (sequence) ¨C Sequence of standard deviations for each channel
        # The way to calculate mean and std for each channel of a image dataset is complicated
        # and thus here, I use the results from ImageNet data which can be accessed here http://image-net.org
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test_u6799959': transforms.Compose([
        transforms.CenterCrop(576),
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = '.\\faces-emotion_v2'

# see https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder for details
# the filefolders in root folder should be like
# root/Angry/xxx.png, root/Angry/xxy.png, root/Fear/123.png, root/Fear/321.png
# it will automatically attach labels to the images that from the corresponding folder
# create a dataset for training and another dataset for testing
image_datasets = {x: datasets.ImageFolder(root = os.path.join(data_dir, x),
                                          transform= data_transforms[x])
                  for x in ['train_u6799959', 'test_u6799959']}

# create a dataloader for training and another dataloader for testing
# shuffle = True makes a mixture of seven face emotions in data loading
# also, as both the images and sample number are large, we use a mini-batch training
dataloaders = {x: torch.utils.data.DataLoader(dataset = image_datasets[x], batch_size=5,
                                             shuffle=True)
              for x in ['train_u6799959', 'test_u6799959']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train_u6799959', 'test_u6799959']}

# class labels, which will be used later as indexes
class_names = image_datasets['train_u6799959'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Now, visualize a few images

# Get a batch of training data
# inputs are tensors which represent a batch of training data
# classes are the corresponding labels in binary form as we are doing a binary classification
inputs, classes = next(iter(dataloaders['train_u6799959']))
inputs.shape # see the shape of inputs. There are two samples, each has three channels with 299*299 pixels each channel

# Make a grid from batch, facilitate batch visualization. Set padding to 2.
out = torchvision.utils.make_grid(tensor=inputs, padding=2)
out.shape

def imshow(inp, title=None):
    """Imshow for Tensor."""
    # transform the shape of input from ([3, 303, 604]) to (303, 604, 3)
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean # denormalize the image data
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

plt.ion()
imshow(out, title=[class_names[x] for x in classes])


'''------------------------------------------------------------------------------'''
'''Now, implement model training'''
''' scheduler is an Learning Rate scheduler object from torch.optim.lr_scheduler,
so we can adjust our learning rate during our training'''

def train_model(model, criterion, optimizer, scheduler, num_epochs=20):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_record = []
    test_loss_record = []

    # training/testing phase, calculate loss and accuracy, update weights for each epoch
    # and find the best model which gives the best accuracy
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        # first train over all training data, then test over all test data
        for phase in ['train_u6799959', 'test_u6799959']:

            if phase == 'train_u6799959':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0  # the total loss for current training/testing epoch
            running_corrects = 0  # the total correct predictions for current training/testing epoch

            # Iterate over training/testing data.
            for _, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train_u6799959'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train_u6799959':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train_u6799959':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase] # averaged loss for current epoch
            epoch_acc = running_corrects.double() / dataset_sizes[phase] # averaged accuracy for current epoch

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'train_u6799959':
                train_loss_record.append(epoch_loss)
            else:
                test_loss_record.append(epoch_loss)

            # deep copy the model if it has the best testing accuracy
            if phase == 'test_u6799959' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_record, test_loss_record


# Load a pretrained model and reset final fully connected layer. In this case I use ResNet
# see https://pytorch.org/docs/stable/torchvision/models.html for more details

'''----------Try different models-----------------'''
'''Model 1: ResNet18 with fixed inner parameters '''
model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)

'''Model 2: ResNet18 with unfixed inner parameters'''
# model_ft = models.resnet18(pretrained=True)
#
# num_ftrs = model_ft.fc.in_features
# model_ft.fc = nn.Linear(num_ftrs, len(class_names))  # Custom the fully connected layer structure of the ResNet model
# model_ft = model_ft.to(device)

'''Model 3: VGG19 with fixed inner parameters'''
'''WARNING: Don't try the following model if your VRAM is not large enough'''
# model_ft = models.vgg19(pretrained=True)
# print(model_ft.classifier[6].out_features) # 1000
# # Freeze training for all layers
# for param in model_ft.features.parameters():
#     param.require_grad = False
# # Newly created modules have require_grad=True by default
# num_features =model_ft.classifier[6].in_features
# features = list(model_ft.classifier.children())[:-1] # Remove last layer
# features.extend([nn.Linear(num_features, len(class_names))]) # Add our layer with 7 outputs
# model_ft.classifier = nn.Sequential(*features) # Replace the model classifier
# print(model_ft)
# model_ft.to(device)

'''-----------------END OF MODEL SELECTION-------------------'''


criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
# Decay LR by a factor of gamma every 5 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)
epoch_num = 50

# Train the model and evaluate by visualize loss variation and
# build confusion matrix for training and testing and visualize precision, recall and specificity for each class
model_ft, train_loss, test_loss = train_model(model = model_ft, criterion = criterion,
                       optimizer = optimizer_ft, scheduler = exp_lr_scheduler, num_epochs=epoch_num)



'''------------------------------------------------------------------------------'''
'''After model training, we can visualize some training performance statistics'''
# plot the loss variation
fig, axs = plt.subplots(1, 2, figsize = (10,5))
axs[0].plot(train_loss)
axs[0].set_title('training loss progress')
axs[0].set(xlabel='training process', ylabel='total loss')
axs[1].plot(test_loss)
axs[1].set_title('testing loss progress')
axs[1].set(xlabel='testing process', ylabel='total loss')
plt.show()

# define function to generate confusion matrix with precision, recall and specificity given the trained model
def model_evaluation(model, phase):
    confusion_matrix = torch.zeros(len(class_names), len(class_names))

    for i, (inputs, labels) in enumerate(dataloaders[phase]):
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        for index in range(preds.size(0)):
            actual_class = labels[index].long()
            predicted_class = preds[index].long()
            confusion_matrix[actual_class][predicted_class] += 1

    precision_list = []
    recall_list = []
    specificity_list = []
    total_number = confusion_matrix.sum().numpy()

    for i in range(len(class_names)):
        tp = confusion_matrix[i][i].numpy() # true positive
        fn = sum(confusion_matrix[i]).numpy() - tp # false negative
        fp = sum(confusion_matrix[:, i]).numpy() - tp # false positive
        tn = total_number - tp - fn - fp # true negative
        if (tp+fp) != 0:
            precision_list.append(tp/(tp+fp))
        else:
            precision_list.append(0)
        if (tp+fn) != 0:
            recall_list.append(tp/(tp+fn))
        else:
            recall_list.append(0)
        specificity_list.append(tn/(tn+fp))

    return confusion_matrix, precision_list, recall_list, specificity_list

train_confu_mat, train_prec, train_recall, train_spec = model_evaluation(model_ft, 'train_u6799959')
test_confu_mat, test_prec, test_recall, test_spec = model_evaluation(model_ft,'test_u6799959')

# plot the statistics of all seven face emotions for both training and testing data
x = np.arange(len(class_names))
auto_width = 0.2

fig = plt.figure(figsize=(10,12))
ax_1 = fig.add_subplot(211)
ax_1.bar(x-auto_width, train_prec, width = auto_width, color='r', label='precision')
ax_1.bar(x, train_recall, width = auto_width, color='b', label = 'recall')
ax_1.bar(x+auto_width, train_spec, width = auto_width, color='g', label = 'specificity')
ax_1.set_ylabel('rate')
ax_1.set_title('TRAINING precision/recall/specificity by expression class')
ax_1.set_xticks(x)
ax_1.set_xticklabels(class_names)
ax_1.legend()

ax_2 = fig.add_subplot(212)
ax_2.bar(x-auto_width, test_prec, width = auto_width, color='r', label='precision')
ax_2.bar(x, test_recall, width = auto_width, color='b', label = 'recall')
ax_2.bar(x+auto_width, test_spec, width = auto_width, color='g', label = 'specificity')
ax_2.set_ylabel('rate')
ax_2.set_title('TESTING precision/recall/specificity by expression class')
ax_2.set_xticks(x)
ax_2.set_xticklabels(class_names)
ax_2.legend()
plt.show()


'''------------------------------------------------------------------------------'''
'''Finally, Visualize the results of model training'''
def visualize_model(model, num_images=8):
    was_training = model.training # a boolean
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['test_u6799959']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('true: {}, predicted: {}'.format(class_names[labels[j]],
                                                              class_names[preds[j]])) # show the predicted label
                imshow(inputs.cpu().data[j]) # shape of (3,299,299)

                if images_so_far == num_images: # plot the first 8 validation images in testing dataloader
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

visualize_model(model_ft)

with torch.no_grad(): # empty GPU cache for cross valiation
    torch.cuda.empty_cache()

'''------------------------------------------------------------------------------'''
'''------------------------------------------------------------------------------'''
'''Part2: Implement Cross-validation to evaluate the performance of tuned model 
more objectively'''
'''First, Construct five training dataloaders and the corresponding validation loaders '''
def split_data(unsplit_datasets, folder = 5):
    '''This function is used to split the original dataset into different parts and
    construct dataloaders'''
    sample_num = len(unsplit_datasets)
    lengths = [int(sample_num / folder)] * (folder - 1)
    lengths.append(sample_num - sum(lengths))

    # use the random_split function https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
    split_datasets = torch.utils.data.random_split(unsplit_datasets,
                                                   lengths=lengths)

    train_dataloader_dict = {} # dictionary to store all the training dataloaders
    valid_datalaoder_dict = {} # all the validation dataloaders

    for i in range(folder):
        current_train_folders = list(range(folder))
        current_train_folders.remove(i)
        current_train_indice = []

        for j in current_train_folders:
            current_train_indice += split_datasets[j].indices

        current_train_subset = torch.utils.data.Subset(dataset=unsplit_datasets,
                                                       indices=current_train_indice)
        current_valid_subset = torch.utils.data.Subset(dataset=unsplit_datasets,
                                                       indices=split_datasets[i].indices)
        current_train_dataloader = torch.utils.data.DataLoader(dataset=current_train_subset,
                                                               batch_size=5, shuffle=True)
        current_valid_dataloader = torch.utils.data.DataLoader(dataset=current_valid_subset,
                                                               batch_size=5, shuffle=True)
        train_dataloader_dict['train' + str(i)] = current_train_dataloader
        valid_datalaoder_dict['valid' + str(i)] = current_valid_dataloader

    return train_dataloader_dict, valid_datalaoder_dict

data_transforms_cv = transforms.Compose([
        # implement center crop and resize the image size for training dataset to 299 pixels
        transforms.CenterCrop(size = 576),
        transforms.Resize(size = 299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir_cv = '.\\faces-emotion_v2\\Subset For Assignment SFEW'

unsplit_datasets = datasets.ImageFolder(root = data_dir_cv,
                                    transform= data_transforms_cv)

train_loader_dict, valid_loader_dict = split_data(unsplit_datasets)


'''------------------------------------------------------------------------------'''
'''Now we want to train the model for each training set and evaluate model on each validation set'''

def train_and_validate_model_cv(train_loader_dict, valid_loader_dict, num_epochs=50, folder = 5):

    since = time.time()
    sample_per_fold = int(len(unsplit_datasets)/folder)
    confusion_matrix = torch.zeros(len(class_names), len(class_names))

    for i in range(folder):
        # reconstruct model for each training set
        model_ft = models.resnet18(pretrained=True)
        for param in model_ft.parameters():
            param.requires_grad = False
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, len(class_names))
        model_ft = model_ft.to(device)

        best_model_wts = copy.deepcopy(model_ft.state_dict())
        best_acc = 0.0

        phase_train = 'train' + str(i)
        phase_valid = 'valid' + str(i)

        criterion = nn.CrossEntropyLoss()
        # Observe that all parameters are being optimized
        optimizer = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.7)
        # Decay LR by a factor of gamma every 5 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)



        # training/testing phase, calculate loss and accuracy, update weights for each epoch
        # and find the best model which gives the best accuracy
        for epoch in range(num_epochs):
            print('Cross-Valid Index {}, Epoch {}/{}'.format(i, epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            # first train over all training data, then test over all test data
            for phase in [phase_train, phase_valid]:

                if phase == phase_train:
                    model_ft.train()  # Set model to training mode
                    dataloaders = train_loader_dict[phase]
                else:
                    model_ft.eval()   # Set model to evaluate mode
                    dataloaders = valid_loader_dict[phase]

                running_loss = 0.0  # the total loss for current training/validation epoch
                running_corrects = 0  # the total correct predictions for current training/validation epoch

                # Iterate over training/validation data.
                for _, (inputs, labels) in enumerate(dataloaders):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward calculation
                    # track history if only in train

                    #with torch.set_grad_enabled(phase == phase_train):
                    outputs = model_ft(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == phase_train:
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == phase_train:
                    scheduler.step()
                    epoch_loss = running_loss / (sample_per_fold*(folder -1))  # averaged loss for current epoch
                    epoch_acc = running_corrects.double() / (sample_per_fold*(folder -1)) # averaged accuracy for current epoch
                else:
                    epoch_loss = running_loss / sample_per_fold # averaged loss for current epoch
                    epoch_acc = running_corrects.double() / sample_per_fold  # averaged accuracy for current epoch

                print('folder{},  {},  Loss: {:.4f} Acc: {:.4f}'.format(
                    i, phase, epoch_loss, epoch_acc))

                # deep copy the model if it has the best testing accuracy
                if phase == phase_valid and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model_ft.state_dict())

            print()

        # load best model weights
        model_ft.load_state_dict(best_model_wts)

        # update the confusion matrix using best model of current validation set
        for _, (inputs, labels) in enumerate(valid_loader_dict[phase_valid]):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            for index in range(preds.size(0)):
                actual_class = labels[index].long()
                predicted_class = preds[index].long()
                confusion_matrix[actual_class][predicted_class] += 1

        with torch.no_grad():
            torch.cuda.empty_cache() # empty cuda cache for next model's GPU training


    time_elapsed = time.time() - since
    print('Training-validation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best test Acc: {:4f}'.format(best_acc))

    return confusion_matrix

# calculate the overall confusion matrix for all samples in validation datasets use above function
cv_confusion_matrix = train_and_validate_model_cv(train_loader_dict, valid_loader_dict)

# visualize precision, recall specificity and overall accuracy
cv_precision_list = []
cv_recall_list = []
cv_specificity_list = []
cv_total_number = cv_confusion_matrix.numpy().sum()
cv_correct_number = torch.diag(cv_confusion_matrix).numpy().sum()
print('Overall Cross Validation Accuracy is: {:.4f} %'.format((cv_correct_number/cv_total_number)*100))

for i in range(len(class_names)):
    tp = cv_confusion_matrix[i][i].numpy() # true positive
    fn = sum(cv_confusion_matrix[i]).numpy() - tp # false negative
    fp = sum(cv_confusion_matrix[:, i]).numpy() - tp # false positive
    tn = cv_total_number - tp - fn - fp # true negative
    if (tp+fp) != 0:
        cv_precision_list.append(tp/(tp+fp))
    else:
        cv_precision_list.append(0)
    if (tp+fn) != 0:
        cv_recall_list.append(tp/(tp+fn))
    else:
        cv_recall_list.append(0)
    cv_specificity_list.append(tn/(tn+fp))

# plot the statistics of all seven face emotions for both training and testing data
x = np.arange(len(class_names))
auto_width = 0.2

fig = plt.figure(figsize=(10,6))
ax_1 = fig.add_subplot(111)
ax_1.bar(x-auto_width, cv_precision_list, width = auto_width, color='r', label='precision')
ax_1.bar(x, cv_recall_list, width = auto_width, color='b', label = 'recall')
ax_1.bar(x+auto_width, cv_specificity_list, width = auto_width, color='g', label = 'specificity')
ax_1.set_ylabel('rate')
ax_1.set_title('Cross Validation precision/recall/specificity by expression class')
ax_1.set_xticks(x)
ax_1.set_xticklabels(class_names)
ax_1.legend()
plt.show()
