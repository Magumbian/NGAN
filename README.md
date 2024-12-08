# NGAN
This is the repository for the paper "N for Parameter: Efficient Neural Audio Super resolution with GAN" 
To train own model with fixed loss weights run the file model.py
To train  own model with load balancer from ENCODEC paper run the file load_balanced_model.py
To run test script run the file test_gpu.py
Update lines 34, 55 and 60 in test_gpu.py to change input sample rate
For details of settings see the repository https://github.com/haoheliu/ssr_eval/tree/main

Code Acknowledgments

We have resused code from the following repositories
1. https://github.com/haoheliu/ssr_eval/tree/main
2. https://github.com/ZhikangNiu/encodec-pytorch
3. https://github.com/kaiidams/soundstream-pytorch
