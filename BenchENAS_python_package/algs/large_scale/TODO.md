逻辑上没啥大问题，为了能够代码整合进去，保持尽可能一致的风格，稍微修改下你的代码哈。

[x] 参考eacnn下的evaluate的代码，把main中的evaluate挪到这个evaluate.py中

[x] 参考eacnn，选用系统弄好的log

[ ] 参考eacnn，对于一些你写死的图像的输入以及分类的个数，通过读取配置文件获取

[x] 把cifar10.py命名也根据eacnn的做出修改，建议用'''备注，因为你这个py脚本打开编辑器会报错

[x] 不要出现print这种debug信息，非必要的都打debug，正常流程可以info级别。