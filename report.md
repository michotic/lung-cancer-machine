# Project Summary

- Got a dataset of CT scan images of the torso region with 4 different labels corresponding to no cancer & 3 types of cancer from almighty Kaggle
- Imported a Vision Transformer (ViT) pre-trained model on image classifcation for its weights and architecture
- Removed its activation layer for 1000 labels and added my own based on my dataset
- Pre-processed my dataset to fit the input layer of my base model
- Fine-tuned the ViT model with my training dataset
- Tested various sets of parameters on the initial training from the base ViT model to see how the model is able to adapt to the new task
- Had a lot of fun

# Dataset

## Link

https://www.kaggle.com/datasets/mohamedhanyyy/chest-ctscan-images

## Number of Samples: 1000

## Dataset Description

Consists of a bunch of CT scans of the chest region with labels corresponding to no cancer, adenocarcinoma, large cell carcinoma, and squamous cell carcinoma. (Various types of lung cancers.)

# Detailed Approach

**Vision Transformer (ViT):** https://huggingface.co/docs/transformers/model_doc/vit

- I found the CT scan dataset on Kaggle because I figure that there's big opportunities in applying machine learning to medical scanning technology, so I was curious to see what I could do with little to no medical-related knowledge outside of a bioinformatics course I took last year.
- Converted the whole dataset to JPEG's because my image processor and model enjoyed "RGB" much more than "RGBA"
- I chose the ViT model for the task of diagnosing multiple types of cancer due because the model architecture and pre-trained weights seemed like they could be valuable to the task at hand. It's an encoder-decoder model that processes images in a bunch of small patches. Considering cancer can appear in the form of small patches and be found in various locations within the body, it seemed like there could be potential for the encoder-decoder architecture in being applied to this task.
- To transfer over the weights to my own model, I used HuggingFace Transformers to fetch the weights & architecture of the Google ViT model. I then replaced it's activation layer with my own, as well as replacing its id2label dict with one for my dataset.
- I then implemented code to save the states of my model after further training so I could save decent ones to revisit. I also added some code to test out my model's overall accuracy on my validation dataset.
- Performance varied quite a bit. Well able to attain peak accuracies of 91% on a testing dataset of around 300 CT scans, and 88% on my separate smaller validation dataset of around 70 samples.
- My "best" model so far in my opinion had the following accuracies on each of it's classification tasks

| Label                   | Right | Wrong | Accuracy |
| ----------------------- | ----- | ----- | -------- |
| adeno carcinoma         | 18    | 5     | 0.78     |
| large cell carcinoma    | 18    | 3     | 0.85     |
| normal                  | 13    | 0     | 1.00     |
| squamous cell carcinoma | 15    | 0     | 1.00     |
| **overall**             | 64    | 8     | 0.88     |

- I hope to continue furthering this model as a side project but I am happy in the progress I have made to develop ML models intended to detect cancer.
