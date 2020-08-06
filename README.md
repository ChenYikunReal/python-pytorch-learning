# PyTorch框架学习记录

![在这里插入图片描述](https://github.com/ChenYikunReal/pytorch_learning/blob/master/images/deep_learning.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)

## 安装Pytorch
安装时间：2020-06-25 ~ 2020-06-29 <br/>
环境：
- Python3.7（可以使用Anaconda自带的Python3.7）
- Anaconda新版（环境变量要配置好）
- Windows10（我还没用Linux装过PyTorch）
- NVIDIA驱动程序和CUDA（只用CPU也行，但我的安装是GPU版本的）

官网：[官网链接](https://pytorch.org/)<br/>
下图是我的显卡配置：<br/>
![在这里插入图片描述](https://github.com/ChenYikunReal/pytorch_learning/blob/master/images/显卡.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)
<br/>根据CUDA版本的对应，以及考虑到现在已经不区分python版本，所以官网给出的conda下载命令如下：<br/>
![在这里插入图片描述](https://github.com/ChenYikunReal/pytorch_learning/blob/master/images/官网conda版本.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)
<br/>同理，官网给出的pip下载命令如下：<br/>
![在这里插入图片描述](https://github.com/ChenYikunReal/pytorch_learning/blob/master/images/官网pip版本.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)
<br/>先说conda吧<br/>
使用<code>conda install pytorch torchvision cudatoolkit=10.2 -c pytorch</code>的话，是从外网上下载的，有墙所以一般会HTTP报错导致下载失败。<br/>
有说法是换成[清华镜像](https://mirrors.tuna.tsinghua.edu.cn/)，给大家分享一下我的配置（user目录下的.condarc文件，找不到的话试试是不是隐藏文件，如果没有可以考虑新建一个吧，新建的时候注意命名是<code>.condarc.</code>，否则Windows系统不允许那种命名格式）。<br/>
我的配置参考了官网的[demo](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/)，鉴于下载的时候路径报错：<code>http://mirrors.tuna.tsinghua.edu.cn/anaconda/http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/win-64</code>，尽管我不知道为什么，但我调整了一下，还说的过去，起码不是路径直接不可访问。<br/>
有一说是不能写https而必须写http，反正两个我都试了，都不行，大家看自己的情况来吧。<br/>
```text
channels:
  - defaults
show_channel_urls: true
channel_alias: http://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - /pkgs/main
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
channel_priority: flexible
envs_dirs:
  - D:\Python\Anaconda\envs                  
pkgs_dirs:
  - D:\Python\Anaconda\pkgs
```
[这篇简书博客](https://www.jianshu.com/p/67981914f365)可能会对你理解上面的一些配置有帮助<br/>
因为下载可能有SSL问题，所以可以使用<code>--trusted-host</code><br/>
我下载的时候一直说1.5.1版本与我本机不匹配，我降低版本也还不被允许，我不指定版本还不能解决，查了很多资料也并不能帮我解决问题，我枯萎。<br/>
因为回避不了<code>CondaHTTPError: HTTP 000 CONNECTION FAILED for url \<http://mirrors.tuna.tsinghua.edu.cn/anaconda...\></code>，使得我不得不放弃清华镜像<br/>
网上也有人说[清华镜像不再可用](https://mirrors.tuna.tsinghua.edu.cn/news/close-anaconda-service/)，但人家早就[恢复运营](https://mirrors.tuna.tsinghua.edu.cn/news/anaconda-restored/)了啊……可能是我自己的问题吧<br/>
网上还有说可以用<code>pip install torch -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com</code>来下载的说法，我试了试，能下载一部分，但还是不能成功。<br/>
有趣的是过程中我发现好像我的Anaconda和Python都是32bit的，然而好像Pytorch要求的是64bit……所以我不得不卸载Anaconda，重新安装64bit，重新配置环境，重新把所有的坑走一遍，然后还是失败……<br/>
我甚至一度怀疑NVIDIA驱动问题，因为我的控制面板里的NVIDIA控制面板是白色的，点击说是“找不到应用程序”，但前面我确实能打开这个东西，只是控制面板找不到，查了很多资料，做了很多尝试，还是不能解决这个空白的问题，后来觉得可能也不是这里的问题。<br/>
![在这里插入图片描述](https://github.com/ChenYikunReal/pytorch_learning/blob/master/images/NVIDIA控制面板.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)
<br/>最后我在[Whl下载页](https://download.pytorch.org/whl/torch_stable.html)这里下载了对应的版本的whl（cu102/torch-1.5.1-cp37-cp37m-win_amd64.whl，页面最底部）：<br/>
![在这里插入图片描述](https://github.com/ChenYikunReal/pytorch_learning/blob/master/images/whl页面1.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)
<br/>这个文件我下过好几次，之前下载好了但都安装失败了，最后干脆下不下来，找了朋友帮忙弄下来的，大概快900MB，本地找个地方存一下，cd命令找到目录，然后<code>pip install cu102/torch-1.5.1-cp37-cp37m-win_amd64.whl</code>，就成功了！！！<br/>
后来发现torchvision库并没有下载下来，使用pip或conda依旧不能成功，还是上面说的下载页底部：<br/>
![在这里插入图片描述](https://github.com/ChenYikunReal/pytorch_learning/blob/master/images/whl页面2.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)
<br/>使用命令<code>pip install torchvision-0.6.1-cp37-cp37m-win_amd64.whl</code>完成下载：<br/>
![在这里插入图片描述](https://github.com/ChenYikunReal/pytorch_learning/blob/master/images/成功下载torchvision.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg5NjMxOA==,size_16,color_FFFFFF,t_70)
<br/>也蛮意外的，经过了五天的努力，我装上了一个PyTorch，泪奔~~~毕竟朋友都很顺利的呢……<br/>
好啦，就总结到这里啦，还望对看到的人有所帮助！Lucky!

## 安装TensorFlow
很多现成的code都是基于TensorFlow写的，下载TensorFlow的命令：<code>pip install tensorflow -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com</code><br/>
我下载的时候本地配的清华镜像不好使，生活不易啊！

# 入门
https://pytorch-cn.readthedocs.io/zh/latest/

## 机器学习
- [深度学习原理](https://www.cntofu.com/book/85/index.html)

## 文本生成
- [文本生成论文集](https://www.cnblogs.com/zjgtan/p/6708468.html)
- [论文检索AMiner](https://www.aminer.cn/)
- [我爱自然语言处理](https://www.52nlp.cn)
- [检索源码Github](https://github.com/)
- [哟林小平-知乎](https://www.zhihu.com/people/lin-zhen-kun-4/posts)
### 2020-07-31
- [7篇NIPS2019文本生成论文推荐](https://zhuanlan.zhihu.com/p/98385473)
    - Kernelized Bayesian Softmax for Text Generation(用于文本生成的核化贝叶斯Softmax)
- [6篇文本生成论文推荐](https://www.jiqizhixin.com/articles/2019-02-21-21)
- [数据到文本生成的近期优质论文解读-微软亚洲](https://zhuanlan.zhihu.com/p/57709494)

## {文件夹: 论文} [文本生成]
- <code>ker_bs</code>：<code>Kernelized Bayesian Softmax for Text Generation</code>
- <code>seq_gan</code>：<code>SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient</code>
