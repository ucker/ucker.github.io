---
layout: post
title:  "The Usage of Git"
date:   2018-02-16
categories: blog
---
# Git的用法

> 请先安装好[git](https://git-scm.com/downloads)，在powershell中进行操作

1. 设置自己的信息

```bash
git config --global user.mail "You Email"
git config --global user.name "Your Name"
```

2. 将代码仓库clone到本地
```bash
git clone https://github.com/ucker/ucker.github.io.git
```

3. 在本地编辑文件，修改完成之后
```bash
git add *
git commit -m "Your Comment"
git push origin master
```

4. 如果已经clone了这个项目，并且需要更新
```bash
git pull
```

5. 如果想新建一个分支
```bash
git checkout -b <new_branch_name>
```

6. 如果想删除这个分支
```bash
git checkout -d <branch_name>
```

7. 如果想切换到分支
```bash
git checkout <branch_name>
```

8. 如果想把更改推送到分支
```bash
git push origin <branch_name>
```

9. 如果想要查看更改历史
```bash
git log
```
