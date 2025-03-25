This is so far just a general outline and idea.
# Voicer
## Basic outline of this project
The main idea is to allow machine read text based on the content with a tone, such as excited, happy, sad, angry, and so on. By providing a text, with context, preferably, outputting a voice with appropriate tone. At the end, may be trying to create a simple website to launch it. 
## How am I thinking about building it
The dataset that I am thinking is to crop faces from movies, and by using CV to capture faces and read their emotions. After that, by matching their emotions with the lines they are speaking, learn and generate a model based on general gender. I am assuming the tone of any emotion is the same for whatever gender. 
## Potential of this project
Instead of hearing from AI, such as ChatGPT and Grok, hearing it with emotions would make it sound more like human. The idea is kind of stupid, but it is using what I want to learn and enhance. 

## Timeline and structures

### Find dataset
An alternative to movies is to use scripts for voice actors or plays.
ost of the datasets that I find is using short audios without prior context. 
Going to try with one movie, harry potter first. Just came to point that the presenting facial recognition is not very accurate, might just going to analysis the tone of each line and discard emotions as it might be a constraint. But will see how it performs after trying. 
### Find research and structures or existing models to capture faces with emotions
SE ResNet model to analysis emotions of face. 
load existing model using: https://paperswithcode.com/model/se-resnet?variant=seresnet50#:~:text=SE%20ResNet%20is%20a%20variant,dynamic%20channel%2Dwise%20feature%20recalibration.
```
import timm
m = timm.create_model('seresnet50', pretrained=True)
m.eval()
```
#### What is next 
Allow real time face capture and emotion analysis for videos, and output lines with emotions. 
##### How
Analysis whenever there are voices appearing, starting capturing faces corresponding to the character, analysis general or overall emotions, and export the line and emotions as csv file for model training. 
### Construct dataset with matching emotions and lines 
