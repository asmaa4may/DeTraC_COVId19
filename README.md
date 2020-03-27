#   DeTraC_COVId19

##        DeTraC COVID-19 classification in chest X-ray images 

*Asmaa Abbas, Mohammed M. Abdelsamea, Mohamed Medhat Gaber21Mathematics **"Classification of COVID-19 in chest X-ray images usingDeTraC 
deep convolutional neural network"**,2020*

### Motivation
 In the last few months, World Health Organization (WHO) has declared that a new virus called COVID-19 has been spread aggressively
 in several countries around the world. The Diagnosis of COVID-19 is typically associated to both the symptoms of pneumonia and
 Chest X-ray tests. Due to the high availability of large-scale annotated imagedatasets, great success has been achieved using convolutional neural
 networks (CNNs)for image recognition and classification. CNN based on transfer learning usually provides an effective solution with
 the limited availability of annotated images by transferring knowledge from pre-trained CNNs (that have been learned from a bench-marked large-scale image dataset) to the 
 specific medical imaging task. Transfer knowledge can be further accomplished by three main scenarios:  shallow-tuning, fine-tuning, 
 or deep-tuning. In this paper, we validate and adopt our previously developed CNN, called Decompose, Transfer, and Compose (DeTraC), for the classification of COVID-19 chest X-ray images.
 DeTraC can deal with any irregularities in the image dataset by investigating its class boundaries using a class decomposition mechanism.
 DeTraCmodel consists of three phases. In the first phase, we train the backbone pre-trained CNN model of DeTraC to extract deep
 local features from each images. Then we apply the class-decomposition layer of DeTraC to simplify the local structure of the data distribution. 
 Finally, we use the class-composition layer of DeTraC to refine the final classification of the images. 
 
 
## **Dataset description**

We used 80 samples of normal109CXRs (with 4020×4892 pixels) from the Japanese Society of Radiological Technology110(JSRT) [19, 20]
and the following open source chest radiography datasets which contains 105 and 11 samples of COVID-19 and SARS (with 4248×3480 pixels),
respectively. we also applied data augumentation techniques to generate more samples such as: flipping up/down and113right/left, 
translation and rotation using random five different angles. This process resulted in a total 1764 samples.
The DeTraC ResNet18 was experiment based on deep learning strategy. The experimental results showed the capability of DeTraC in the detection of COVID-19 cases from a comprehensive image dataset collected 
 from several hospitals saround the world. A high accuracy of 95.12% (with sensitivity of 97.91%, specificity of91.87%, and precision of
 93.36%) was achieved by DeTraC in the detection of COVID-19 X-ray images from normal, and severe acute respiratory syndrome cases. 
 In addition, we compared the

```python
open source for chest radiography datasets:
 https://github.com/ieee8023/covid-chestxray-dataset
 https://ieeexplore.ieee.org/abstract/document/6663723
 https://ieeexplore.ieee.org/abstract/document/6616679
```

## **Results**

DeTraC_COVID19 achieved high accuracy of 95.12% which proved that CNNs have an effective and robust solution for the detection 
of the COVID-19 cases from CXR images and as a consequence this can be contributed to control the spread of the disease.


**Table 1:** the samples distribution in each class of chest X-ray dataset before and after class decomposition.

<table>
  <tr>
    <th>Original <br>Labels</th>
    <th colspan="2">normal</th>
    <th colspan="2">COVID_19</th>
    <th colspan="2">SARS</th>
  </tr>
  <tr>
    <td># instances</td>
    <td colspan="2">80</td>
    <td colspan="2">105</td>
    <td colspan="2">11</td>
  </tr>
  <tr>
    <td>Decomposed<br>Labels</td>
    <td>norm_1</td>
    <td>norm_2</td>
    <td>COVID19_1</td>
    <td>COVID19_2</td>
    <td>SARS_1</td>
    <td>SARS_2</td>
  </tr>
  <tr>
    <td># instances</td>
    <td>441</td>
    <td>279</td>
    <td>666</td>
    <td>283</td>
    <td>63</td>
    <td>36</td>
  </tr>
</table>

**Table 2:** COVID-19 classification obtained byDeTraC-ResNet18on chest X-rayimages.
|  Accuracy | Sensitivity  |  Specificity |  Precision |
| ------------ | ------------ | ------------ | ------------ |
|  95.12%      | 97.91%      |      91.87%  |  93.36% |

 Fig: the learning curve accuracy and loss between training and test sets.

![1](https://github.com/asmaa4may/DeTraC_COVId19/blob/master/images/Learning%20curve.png ) 



## Contact

 If you would like to contribute DeTraC_COVID-19 x-ray images,please contact us at asmaa.abbas@science.aun.edu.egm 
 and mohammed.abdelsamea@bcu.ac.uk  or mohamed.gaber@bcu.ac.uk

 ## Contribution
 The source code of the DeTraC_COVID19 is available on GitHub in https://github.com/asmaa4may/DeTraC_COVId19.
 
## License
[MIT](https://github.com/asmaa4may/DeTraC_COVId19/blob/master/LICENSE)





