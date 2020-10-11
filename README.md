# MFF-GAN
Code of paper MFF-GAN: An unsupervised generative adversarial network with adaptive and gradient joint constraints for multi-focus image fusion.
````
@article{zhang2020mff,
  title={MFF-GAN: An unsupervised generative adversarial network with adaptive and gradient joint constraints for multi-focus image fusion},
  author={Zhang, Hao and Le, Zhuliang and Shao, Zhenfeng and Xu, Han and Ma, Jiayi},
  journal={Information Fusion},
  year={2020},
  publisher={Elsevier}
}
````

#### Prepare data :<br>
Run "main.m" (the first function) to convert source images from RGB color space to YCbCr.

#### To train :<br>
Put training image pairs (Y channel) in the "Train_near" and "Train_far" folders, and run "CUDA_VISIBLE_DEVICES=0 python main.py" to train the network.

#### To test :<br>
Put test image pairs (Y channel) in the "Test_near" and "Test_far" folders, and run "CUDA_VISIBLE_DEVICES=0 python test.py" to test the trained model.
You can also directly use the trained model we provide.

#### Restore the output of networks to RGB space :<br>
Run "main.m" (the second function) to restore the output of networks to RGB color space.
