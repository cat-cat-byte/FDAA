## FDAA: A Feature Distribution-aware Transferable Adversarial Attack Method 
Tensorflow implementation for "A Feature Distribution-aware Transferable Adversarial Attack Method"

## requirments
- python 3.6.0
- tensorflow 1.14.0
- Numpy 1.19.2
- Keras 2.2.4
- Scipy 1.2.1
- Pillow 8.4.0
- opencv-python 4.5.5.64

## dataset download
Before start the tests, you should download the [ImageNet-compatible](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/dataset), [NT model weight](https://github.com/tensorflow/models/tree/master/research/slim) and the [EAT model weight](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models).Then place these model checkpoint files in ./models_tf direction.

### layer name
- inception_v3(bs,35,35,256):'InceptionV3/InceptionV3/Mixed_5b/concat'
- inception_v4(bs,35,35,384):'InceptionV4/InceptionV4/Mixed_5e/concat'
- inception_resnet_v2(bs,71,71,192):'InceptionResnetV2/InceptionResnetV2/Conv2d_4a_3x3/Relu'
- resnet_v2_152(bs,19,19,512):'resnet_v2_152/block2/unit_8/bottleneck_v2/add'

### introduction for files
- ***attack_method.py***: Integrates various geometric data augmentation adversarial sample algorithms.
- ***attacks.py***: A standard framework for feature-based adversarial attack algorithms. The desired feature-based adversarial attack algorithm can be selected directly through parameters.
- ***FIA.py***: Implementation of the FIA adversarial attack.
- ***NAA.py***: Implementation of the NAA adversarial attack.
- ***RPA.py***: Implementation of the RPA adversarial attack.
- ***verify.py***: Tests the attack success rate of generating adversarial samples.
- ***vision_feature_map.py***: Visualizes intermediate-level features.
- ***get_concat_img.py***: Generates N*M feature carrier images.
- ***HIT.py***: Implementation of the HIT feature insertion adversarial attack.
- ***FDAA.py***: Implementation code for region-based attacks. (Assumes Smap file, adversarial samples for feature disruption algorithm and adversarial samples for HIT feature insertion are prepared.)
- ***utils.py***: Common basic functions used in the above code.
- ***net directory***: Frameworks for commonly used networks.
- ***dataset directory***: Data used for testing, including images from the test set, CAM, Smap, feature maps, and aggregated feature maps.
- ***AFMA.py***:Implementation of the AFMA adversarial attack.

### How to use
* AFMA
    > python AFMA.py
* FDAA
    > before start FDAA, you should make sure that Smap file, adversarial samples for feature disruption algorithm and adversarial samples for HIT feature insertion are prepared. And then run this command:
    python FDAA.py
* NAA.py
    > python NAA.py
* RPA.py
    > python RPA.py