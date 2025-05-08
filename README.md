# FecundityModel 9.8K V1
Model: "sequential"
_____________________________________________________________________________________
 Layer (type)                 Output Shape              Param #       
=====================================================================================
 conv2d (Conv2D)              (None, 73, 73, 32)         896          
 max_pooling2d (MaxPooling2D) (None, 36, 36, 32)         0            
 conv2d_1 (Conv2D)            (None, 34, 34, 64)         18,496       
 max_pooling2d_1 (MaxPooling2D)(None, 17, 17, 64)        0            
 conv2d_2 (Conv2D)            (None, 15, 15, 64)         36,928       
 flatten (Flatten)            (None, 14400)              0            
 dense (Dense)                (None, 64)                 921,664      
 dense_1 (Dense)              (None, 43)                 2,795        
=====================================================================================
 Total params: 980,781 (3.74 MB)  
 Trainable params: 980,779 (3.74 MB)  
 Non-trainable params: 0 (0.00 B)  
 Optimizer params: 2 (12.00 B)


