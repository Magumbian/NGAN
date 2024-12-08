# NGAN
This is the repository for the paper "N for Parameter: Efficient Neural Audio Super resolution with GAN" 
</br>To train own model with fixed loss weights run the file model.py
</br>To train  own model with load balancer from ENCODEC paper run the file load_balanced_model.py
</br>To run test script run the file test_gpu.py
<ul>
<li>Update lines 34, 55 and 60 in test_gpu.py to change input sample rate
<li>For details of settings see the repository https://github.com/haoheliu/ssr_eval/tree/main
</ul>
<h3>Code Acknowledgments</h3>

We have reused code from the following repositories
1. https://github.com/haoheliu/ssr_eval/tree/main
2. https://github.com/ZhikangNiu/encodec-pytorch
3. https://github.com/kaiidams/soundstream-pytorch
