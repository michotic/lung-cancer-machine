# Research Notes

Just to keep track of what I've been doing to avoid repeating past mistakes and help me remember what I did when I have to write the final report.

## Base model: https://huggingface.co/google/vit-base-patch16-224

## Beginning of Research

I started out with a pre-trained Google Vision Transformer (ViT) model that was trained to classify images into 1000 different labels. I figured this would be a good foundation for an image-classification model for tasks where rather small details are of importance based on the fact that the model was intended to classify a relatively large amount of objects.

I started out this research project with the goal of using machine learning to aid in the task of cancer diagnosis. Initially I had used a KNN classifier to diagnose samples of breast tissue tumors as either benign or malignant based upon features consisting of 9 precise physical measurements. It attained high accuracy but in regards to the models we covered throughout the course I did not feel it was adequate as a final research project.

## Research Objective

I have now set out to fine-tine the pre-trained ViT model to the task of classifying CT scans of the torso region into 4 different labels. One label is "normal" meaning non-cancerous, and the other 3 are some types of cancer that can be found within this region of the body. Ideally one day this type of technology could be used to lighten the workload of medical workers if performance were able to reach a level equal, if not better, to humans in regards of consistency and overall accuracy.

## Approach

The approach I have taken to accomplish this task is starting out with the base Google ViT model. I then gutted it's output label dictionary because I'm not here to classify different types of animals and other objects. I then added in my own output labels along with id2label and label2id maps based on the end goal. After this I now had a model consisting of all the initial weights of the ViT model, with my own activation layer for 4 labels as the output layer.

I then had to write some code to get my images into the format the ViT model was built to accept, which just meant resizing them to 226x226 and creating tensors of their RGB values. I then ran my dataset through my preprocessing code and made fixes until everything was formatted as intended, and able to be inputted into my model for a training run without any errors.

### Training

Overfitting ended up rendering the first bit of training as a bit of a waste. Scroll down to round 2 for final results.

After I had my model architecture and dataset in a useable state, I then added some code to save the state of my models weights at each epoch at the end of a training run. Initial accuracy in training was ~58% This allowed me to take the weights and overall state of my model from the best epoch of my last training run, and use that as the starting state for future training runs. After some more training runs and upping the epochs/run from 3 to 6, I had attained 83.17% accuracy in training by this point. To help avoid over-fitting while still being able to train a bit more on the same data, I added further pre-processing to the training images for future training. The new transformation was a random crop of the image that would keep between 75-100% of the original image. I figured this could have some reasoning behind it as people do come in various shapes and sizes, so adding some variety to the scale and placement of areas of interest could be helpful in testing on unseen scans.

After getting up to consistent >88% accuracy with a loss value of 0.4%, I lowered the # of epochs down to 4 per training cycle, as I found the later epochs would maintain stagnant accuracy and primarily reduce the loss metric more than anything, and hopefully more cycles of this lower # of epochs will allow for more of the significant changes to the weights that occur earlier-on in each training cycle. On top of this I added a weight decay between 0.025-0.1 for a few training cycles to help break down pathways that weren't seeing much usage, in hindsight this may have been best-implemented earlier on in this research as the model probably included some sub-optimal pathways formed from the pre-trained weights provided with the initial base ViT model.

The cycles with increased weight decay displayed more volatility in their performance, with initial accuracies as low as 66%, but ramping up to accuracies in the low 90%'s with near zero loss values. Increasing the number of epochs again may help with the volatility introduced by the weight decay and will probably be the last changes I attempt to squeeze better performance out of this dataset + model pairing.

After validating the data there was definitely some extreme overfitting to the training data. Results in validation were in the 20% area. Will be scratching weights and fine-tuning from the initial base ViT model again.

##Round 2
After the first 3 epochs of training with new parameters and a fresh base model, I attained 65% accuracy with the testing set and 37.5% accuracy with the validation set, mariginally better. Validation is now being done at the end of each set of training epochs to hopefully avoid over-fitting more carefully.

Up to 86.9% accuracy in training and 42% accuracy on an external validation dataset.

###Latest performance on validation set:
adeno carcinoma
right: 0 wrong: 23
accuracy: 0.0

large cell carcinoma
right: 8 wrong: 13
accuracy: 0.38095238095238093

normal
right: 12 wrong: 1
accuracy: 0.9230769230769231

squamous cell carcinoma
right: 10 wrong: 5
accuracy: 0.6666666666666666

overall
right: 30 wrong: 42
accuracy: 0.4166666666666667

The model seems to perform well on physically larger cancers which is not very surprising. The 1 wrong in non-cancerous predictions appeared relatively late in the fine-tuning process but was the most evenly-distributed and highest accuracy performance I have obtained on my validation dataset so far.

### A slightly different approach

Tried a radically different set of parameters when initially training the base-model on the new task. Initial results:

adeno carcinoma
right: 6 wrong: 17
accuracy: 0.2608695652173913

large cell carcinoma
right: 0 wrong: 21
accuracy: 0.0

normal
right: 9 wrong: 4
accuracy: 0.6923076923076923

squamous cell carcinoma
right: 1 wrong: 14
accuracy: 0.06666666666666667

overall
right: 16 wrong: 56
accuracy: 0.2222222222222222

It might be able to recognize types that prior models hadn't, too early to tell.

---

| Label                   | A    | B    | C    | D    | E    | #   |
| ----------------------- | ---- | ---- | ---- | ---- | ---- | --- |
| adeno carcinoma         | 0.78 | 0.78 | 0.69 | 0.86 | 0.86 | 0   |
| large cell carcinoma    | 0.76 | 0.85 | 0.80 | 0.80 | 0.80 | 0   |
| normal                  | 0.92 | 1.00 | 0.92 | 0.84 | 0.92 | 0   |
| squamous cell carcinoma | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0   |
| **overall**             | 0.84 | 0.88 | 0.83 | 0.87 | 0.88 | 0   |
